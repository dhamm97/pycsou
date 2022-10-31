import cmath
import collections
import collections.abc as cabc
import types
import typing as typ
import warnings

import dask
import dask.graph_manipulation as dgm
import finufft
import numba
import numba.cuda
import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw

__all__ = [
    "NUFFT",
]

SignT = typ.Literal[1, -1]
sign_default = 1
eps_default = 1e-4


@pycrt.enforce_precision(i=("z", "beta"))
def ES_kernel(z: pyct.NDArray, beta: pyct.Real) -> pyct.NDArray:
    r"""
    Evaluate the Exponential of Semi-Circle (ES) kernel.

    Parameters
    ----------
    z: pyct.NDArray
        (N,) evaluation points
    beta: pyct.Real
        cutoff-frequency

    Returns
    -------
    phi: pyct.NDArray
        (N,) kernel values at evaluation points.

    Notes
    -----
    The Exponential of Semi-Circle (ES) kernel is defined as (see [FINUFFT]_ eq. (1.8)):

    .. math::

       \phi_\beta(z)
       =
       \begin{cases}
           e^{\beta(\sqrt{1-z^2}-1)}, & |z|\leq 1,\\
           0,                         &\text{otherwise.}
       \end{cases}
    """
    assert beta > 0
    xp = pycu.get_array_module(z)

    phi = xp.zeros_like(z)
    mask = xp.fabs(z) <= 1
    phi[mask] = xp.exp(beta * (xp.sqrt(1 - z[mask] ** 2) - 1))

    return phi


class NUFFT(pyca.LinOp):
    r"""
    Non-Uniform Fast Fourier Transform (NUFFT) of Type 1/2/3 (for :math:`d=\{1,2,3\}`).

    The *Non-Uniform Fast Fourier Transform (NUFFT)* generalizes the FFT to off-grid data.
    There are three types of NUFFTs proposed in the literature:

    * Type 1 (*non-uniform to uniform*),
    * Type 2 (*uniform to non-uniform*),
    * Type 3 (*non-uniform to non-uniform*).

    See the notes below, including [FINUFFT]_, for definitions of the various transforms and
    algorithmic details.

    The transforms should be instantiated via
    :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type1`,
    :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type2`, and
    :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type3` respectively.
    (See each method for usage examples.)

    The dimension of each transform is inferred from the dimensions of the input arguments, with
    support for :math:`d=\{1,2,3\}`.

    Notes
    -----
    We adopt here the same notational conventions as in [FINUFFT]_.

    **Mathematical Definition.**
    Let :math:`d\in\{1,2,3\}` and consider the mesh

    .. math::

       \mathcal{I}_{N_1,\ldots,N_d}
       =
       \mathcal{I}_{N_1} \times \cdots \times \mathcal{I}_{N_d}
       \subset \mathbb{Z}^d,

    where the mesh indices :math:`\mathcal{I}_{N_i}\subset\mathbb{Z}` are given for each dimension
    :math:`i=1,\dots, d` by:

    .. math::

       \mathcal{I}_{N_i}
       =
       \begin{cases}
           [[-N_i/2, N_i/2-1]], & N_i\in 2\mathbb{N} \text{ (even)}, \\
           [[-(N_i-1)/2, (N_i-1)/2]], & N_i\in 2\mathbb{N}+1 \text{ (odd)}.
       \end{cases}


    Then the NUFFT operators approximate, up to a requested relative accuracy
    :math:`\varepsilon \geq 0`, [#]_ the following exponential sums:

    .. math::

       \begin{align}
           (1)\; &u_{\mathbf{n}} = \sum_{j=1}^{M} w_{j} e^{\sigma i\langle \mathbf{n}, \mathbf{x}_{j} \rangle}, \quad &\mathbf{n}\in \mathcal{I}_{N_1,\ldots, N_d},\qquad &\text{Type 1 (non-uniform to uniform)}\\
           (2)\; &w_{j} = \sum_{\mathbf{n}\in\mathcal{I}_{N_1,\ldots, N_d}} u_{\mathbf{n}} e^{\sigma i\langle \mathbf{n}, \mathbf{x}_{j} \rangle }, \quad &j=1,\ldots, M,\qquad  &\text{Type 2 (uniform to non-uniform)}\\
           (3)\; &v_{k} = \sum_{j=1}^{M} w_{j} e^{\sigma i\langle \mathbf{z}_k, \mathbf{x}_{j} \rangle }, \quad &k=1,\ldots, N, \qquad &\text{Type 3 (non-uniform to non-uniform)}
       \end{align}

    where :math:`\sigma \in \{+1, -1\}` defines the sign of the transform and
    :math:`u_{\mathbf{n}}, v_{k}, w_{j}\in \mathbb{C}`.
    For the type-1 and type-2 NUFFTs, the non-uniform samples :math:`\mathbf{x}_{j}` are assumed to
    lie in :math:`[-\pi,\pi)^{d}`.
    For the type-3 NUFFT, the non-uniform samples :math:`\mathbf{x}_{j}` and
    :math:`\mathbf{z}_{k}` are arbitrary points in :math:`\mathbb{R}^{d}`.

    **Adjoint NUFFTs.**
    The type-1 and type-2 NUFFTs with opposite signs form an *adjoint pair*.
    The adjoint of the type-3 NUFFT is obtained by flipping the transform's sign and switching the
    roles of :math:`\mathbf{z}_k` and :math:`\mathbf{x}_{j}` in (3).

    **Lipschitz Constants.**
    We bound the Lipschitz constant by the Frobenius norm of the operators, which yields :math:`L
    \leq \sqrt{NM}`.
    Note that these Lipschitz constants are cheap to compute but may be pessimistic. Tighter
    Lipschitz constants can be computed by calling :py:meth:`~pycsou.abc.operator.LinOp.lipschitz`.

    **Error Analysis.**
    Let :math:`\tilde{\mathbf{u}}\in\mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` and
    :math:`\tilde{\mathbf{w}}\in\mathbb{C}^{M}` be the outputs of the type-1 and type-2 NUFFT
    algorithms which approximate the sequences
    :math:`{\mathbf{u}}\in\mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` and
    :math:`{\mathbf{w}}\in\mathbb{C}^{M}` defined in (1) and (2) respectively.
    Then [FINUFFT]_ shows that the relative errors
    :math:`\|\tilde{\mathbf{u}}-{\mathbf{u}}\|_2/\|{\mathbf{u}}\|_2` and
    :math:`\|\tilde{\mathbf{w}}-{\mathbf{w}}\|_2/\|{\mathbf{w}}\|_2` are **almost always similar to
    the user-requested tolerance** :math:`\varepsilon`, except when round-off error dominates
    (i.e. very small user-requested tolerances).
    The same holds approximately for the type-3 NUFFT.
    Note however that this is a *typical error analysis*: some degenerate (but rare) worst-case
    scenarios can result in higher errors.

    **Complexity.**
    Naive evaluation of the exponential sums (1), (2) and (3) above costs :math:`O(NM)`, where
    :math:`N=N_{1}\ldots N_{d}` for the type-1 and type-2 NUFFTs.
    NUFFT algorithms approximate these sums to a user-specified relative tolerance
    :math:`\varepsilon` in log-linear complexity in :math:`N` and :math:`M`.
    The complexity of the various NUFFTs are given by (see [FINUFFT]_):

    .. math::

       &\mathcal{O}\left(N \log(N) + M|\log(\varepsilon)|^d\right)\qquad &\text{(Type 1 and 2)}\\
       &\mathcal{O}\left(\Pi_{i=1}^dX_iZ_i\sum_{i=1}^d\log(X_iZ_i) + (M + N)|\log(\varepsilon)|^d\right)\qquad &\text{(Type 3)}

    where :math:`X_i = \max_{j=1,\ldots,M}|(\mathbf{x}_{j})_i|` and :math:`Z_i =
    \max_{k=1,\ldots,N}|(\mathbf{z}_k)_i|` for :math:`i=1,\ldots,d`.
    The terms above correspond to the complexities of the FFT and spreading/interpolation steps
    respectively.

    The complexity of the type-3 NUFFT can be arbitrarily large for poorly-centered data.
    An easy fix consists in centering the data before/after the NUFFT via pre/post-phasing
    operations, as described in equation (3.24) of [FINUFFT]_.
    This optimization is automatically carried out by FINUFFT if the compute/memory gains are
    significant.

    **Backend.**
    The NUFFT transforms are computed via Python wrappers to
    `FINUFFT <https://github.com/flatironinstitute/finufft>`_ and
    `cuFINUFFT <https://github.com/flatironinstitute/cufinufft>`_.
    (See also [FINUFFT]_ and [cuFINUFFT]_.)
    These librairies perform the expensive spreading/interpolation between nonuniform points and the
    fine grid via the "exponential of semicircle" kernel.

    **Optional Parameters.**
    [cu]FINUFFT exposes many optional parameters to adjust the performance of the algorithms, change
    the output format, or provide debug/timing information.
    While the default options are sensible for most setups, advanced users may overwrite them via
    the ``kwargs`` parameter of
    :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type1`,
    :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type2`, and
    :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type3`.
    See the `guru interface <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_
    from FINUFFT and its `companion page
    <https://finufft.readthedocs.io/en/latest/opts.html#options-parameters>`_ for details.

    .. Hint::

       FINUFFT exposes a ``dtype`` keyword to control the precision (single or double) at which
       transforms are performed.
       This parameter is ignored by :py:class:`~pycsou.operator.linop.nufft.NUFFT`: use the context
       manager :py:class:`~pycsou.runtime.Precision` to control floating point precision.

    .. Hint::

       The NUFFT is performed in **chunks of (n_trans,)**, where `n_trans` denotes the number
       of simultaneous transforms requested.
       (See the ``n_trans`` parameter of `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.)

       Good performance is obtained when each chunk fits easily in memory. This recommendation also
       applies to Dask inputs which are re-chunked internally to be `n_trans`-compliant.

       This parameter also affects performance of the ``eps=0`` case: increasing ``n_trans`` may
       improve performance when doing several transforms in parallel.

    .. Warning::

       Since FINUFFT plans cannot be shared among different processes, this class is **only
       compatible** with Dask's thread-based schedulers, i.e.:

       * ``scheduler='thread'``
       * ``scheduler='synchronous'``
       * ``distributed.Client(processes=False)``

       Chunks are hence processed serially.
       (Each chunk is multi-threaded however.)

    .. [#] :math:`\varepsilon= 0` means that no approximation is performed: the exponential sums
           are naively computed by direct evaluation.

    See Also
    --------
    FFT, DCT, Radon
    """
    # The goal of this wrapper class is to sanitize __init__() inputs.

    def __init__(self, shape: pyct.OpShape):
        super().__init__(shape=shape)

    @staticmethod
    @pycrt.enforce_precision(i="x", o=False, allow_None=False)
    def type1(
        x: pyct.NDArray,
        N: typ.Union[pyct.Integer, tuple[pyct.Integer, ...]],
        isign: SignT = sign_default,
        eps: pyct.Real = eps_default,
        real: bool = False,
        **kwargs,
    ) -> pyct.OpT:
        r"""
        Type-1 NUFFT (non-uniform to uniform).

        Parameters
        ----------
        x: pyct.NDArray
            (M, [d]) d-dimensional sample points :math:`\mathbf{x}_{j} \in [-\pi,\pi)^{d}`.
        N: int | tuple[int]
            ([d],) mesh size in each dimension :math:`(N_1, \ldots, N_d)`.

            If `N` is an integer, then the mesh is assumed to have the same size in each dimension.
        isign: 1 | -1
            Sign :math:`\sigma` of the transform.
        eps: float
            Requested relative accuracy :math:`\varepsilon \geq 0`.

            If ``eps=0``, the transform is computed exactly via direct evaluation of the exponential
            sum using a Numba JIT-compiled kernel.
        real: bool
            If ``True``, assumes ``.apply()`` takes (..., M) inputs in :math:`\mathbb{R}^{M}`.

            If ``False``, then ``.apply()`` takes (..., 2M) inputs, i.e. :math:`\mathbb{C}^{M}`
            vectors viewed as bijections with :math:`\mathbb{R}^{2M}`.
        **kwargs
            Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
            (Illegal keywords are dropped silently.)
            Most useful are ``n_trans``, ``nthreads`` and ``debug``.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            (2N.prod(), M) or (2N.prod(), 2M) type-1 NUFFT.

        Examples
        --------

        .. code-block:: python3

           import numpy as np
           import pycsou.operator.linop as pycl
           import pycsou.runtime as pycrt
           import pycsou.util as pycu

           rng = np.random.default_rng(0)
           D, M, N = 2, 200, 5  # D denotes the dimension of the data
           x = np.fmod(rng.normal(size=(M, D)), 2 * np.pi)

           with pycrt.Precision(pycrt.Width.SINGLE):
               # The NUFFT dimension (1/2/3) is inferred from the trailing dimension of x.
               # Its precision is controlled by the context manager.
               N_trans = 5
               A = pycl.NUFFT.type1(
                       x, N,
                       n_trans=N_trans,
                       isign=-1,
                       eps=1e-3,
                   )

               # Pycsou operators only support real inputs/outputs, so we use the functions
               # pycu.view_as_[complex/real] to interpret complex arrays as real arrays (and
               # vice-versa).
               arr =        rng.normal(size=(3, N_trans, M)) \
                     + 1j * rng.normal(size=(3, N_trans, M))
               A_out_fw = pycu.view_as_complex(A.apply(pycu.view_as_real(arr)))
               A_out_bw = pycu.view_as_complex(A.adjoint(pycu.view_as_real(A_out_fw)))
        """
        init_kwargs = _NUFFT1._sanitize_init_kwargs(
            x=x,
            N=N,
            isign=isign,
            eps=eps,
            real_in=real,
            real_out=False,
            **kwargs,
        )
        return _NUFFT1(**init_kwargs).squeeze()

    @staticmethod
    @pycrt.enforce_precision(i="x", o=False, allow_None=False)
    def type2(
        x: pyct.NDArray,
        N: typ.Union[pyct.Integer, tuple[pyct.Integer, ...]],
        isign: SignT = sign_default,
        eps: pyct.Real = eps_default,
        real: bool = False,
        **kwargs,
    ) -> pyct.OpT:
        r"""
        Type-2 NUFFT (uniform to non-uniform).

        Parameters
        ----------
        x: pyct.NDArray
            (M, [d]) d-dimensional sample points :math:`\mathbf{x}_{j} \in [-\pi,\pi]^{d}`.
        N: int | tuple[int]
            ([d],) mesh size in each dimension :math:`(N_1, \ldots, N_d)`.

            If `N` is an integer, then the mesh is assumed to have the same size in each dimension.
        isign: 1 | -1
            Sign :math:`\sigma` of the transform.
        eps: float
            Requested relative accuracy :math:`\varepsilon \geq 0`.

            If ``eps=0``, the transform is computed exactly via direct evaluation of the exponential
            sum using a Numba JIT-compiled kernel.
        real: bool
            If ``True``, assumes ``.apply()`` takes (..., N.prod()) inputs in :math:`\mathbb{R}^{N}`.

            If ``False``, then ``.apply()`` takes (..., 2N.prod()) inputs, i.e. :math:`\mathbb{C}^{N}`
            vectors viewed as bijections with :math:`\mathbb{R}^{2N}`.
        **kwargs
            Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
            (Illegal keywords are dropped silently.)
            Most useful are ``n_trans``, ``nthreads`` and ``debug``.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            (2M, N.prod()) or (2M, 2N.prod()) type-2 NUFFT.

        Examples
        --------

        .. code-block:: python3

           import numpy as np
           import pycsou.operator.linop as pycl
           import pycsou.runtime as pycrt
           import pycsou.util as pycu

           rng = np.random.default_rng(0)
           D, M, N = 2, 200, 5  # D denotes the dimension of the data
           N_full = (N,) * D
           x = np.fmod(rng.normal(size=(M, D)), 2 * np.pi)

           with pycrt.Precision(pycrt.Width.SINGLE):
               # The NUFFT dimension (1/2/3) is inferred from the trailing dimension of x.
               # Its precision is controlled by the context manager.
               N_trans = 5
               A = pycl.NUFFT.type2(
                       x, N,
                       n_trans=N_trans,
                       isign=-1,
                       eps=1e-3,
                   )

               # Pycsou operators only support real inputs/outputs, so we use the functions
               # pycu.view_as_[complex/real] to interpret complex arrays as real arrays (and
               # vice-versa).
               arr = np.reshape(
                          rng.normal(size=(3, N_trans, *N_full))
                   + 1j * rng.normal(size=(3, N_trans, *N_full)),
                   (3, N_trans, -1),
               )
               A_out_fw = pycu.view_as_complex(A.apply(pycu.view_as_real(arr)))
               A_out_bw = pycu.view_as_complex(A.adjoint(pycu.view_as_real(A_out_fw)))
        """
        init_kwargs = _NUFFT1._sanitize_init_kwargs(
            x=x,
            N=N,
            isign=-isign,
            eps=eps,
            real_in=False,
            real_out=real,
            **kwargs,
        )
        op_t1 = _NUFFT1(**init_kwargs)
        op_t2 = op_t1.squeeze().T
        op_t2._name = "_NUFFT2"

        # Some methods need to be overloaded (even if non-arithmetic) since NUFFT interprets their
        # calling parameters differently. (Ex: to_sciop's `dtype` discarded at times.)

        def op_to_sciop(_, **kwargs):
            op_s = op_t1.to_sciop(**kwargs)
            return op_s.T

        op_t2.to_sciop = types.MethodType(op_to_sciop, op_t2)  # non-arithmetic method
        op_t2.lipschitz = types.MethodType(NUFFT.lipschitz, op_t1)
        # not strictly necessary, but users will probably want to access it.
        op_t2.params = types.MethodType(_NUFFT1.params, op_t1)
        return op_t2

    @staticmethod
    @pycrt.enforce_precision(i=("x", "z"), o=False, allow_None=False)
    def type3(
        x: pyct.NDArray,
        z: pyct.NDArray,
        isign: SignT = sign_default,
        eps: pyct.Real = eps_default,
        real: bool = False,
        **kwargs,
    ) -> pyct.OpT:
        r"""
        Type-3 NUFFT (non-uniform to non-uniform).

        Parameters
        ----------
        x: pyct.NDArray
            (M, [d]) d-dimensional sample points :math:`\mathbf{x}_{j} \in \mathbb{R}^{d}`.
        z: pyct.NDArray
            (N, [d]) d-dimensional query points :math:`\mathbf{z}_{k} \in \mathbb{R}^{d}`.
        isign: 1 | -1
            Sign :math:`\sigma` of the transform.
        eps: float
            Requested relative accuracy :math:`\varepsilon \geq 0`.

            If ``eps=0``, the transform is computed exactly via direct evaluation of the exponential
            sum using a Numba JIT-compiled kernel.
        real: bool
            If ``True``, assumes ``.apply()`` takes (..., M) inputs in :math:`\mathbb{R}^{M}`.

            If ``False``, then ``.apply()`` takes (..., 2M) inputs, i.e. :math:`\mathbb{C}^{M}`
            vectors viewed as bijections with :math:`\mathbb{R}^{2M}`.
        **kwargs
            Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
            (Illegal keywords are dropped silently.)
            Most useful are ``n_trans``, ``nthreads`` and ``debug``.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            (2N, M) or (2N, 2M) type-3 NUFFT.

        Examples
        --------

        .. code-block:: python3

           import numpy as np
           import pycsou.operator.linop as pycl
           import pycsou.runtime as pycrt
           import pycsou.util as pycu

           rng = np.random.default_rng(0)
           D, M, N = 3, 200, 5  # D denotes the dimension of the data
           x = rng.normal(size=(M, D)) + 2000  # Poorly-centered data
           z = rng.normal(size=(N, D))
           with pycrt.Precision(pycrt.Width.SINGLE):
               # The NUFFT dimension (1/2/3) is inferred from the trailing dimension of x/z.
               # Its precision is controlled by the context manager.
               N_trans = 20
               A = pycl.NUFFT.type3(
                       x, z,
                       n_trans=N_trans,
                       isign=-1,
                       eps=1e-6,
                    )

               # Pycsou operators only support real inputs/outputs, so we use the functions
               # pycu.view_as_[complex/real] to interpret complex arrays as real arrays (and
               # vice-versa).
               arr =        rng.normal(size=(3, N_trans, M)) \
                     + 1j * rng.normal(size=(3, N_trans, M))
               A_out_fw = pycu.view_as_complex(A.apply(pycu.view_as_real(arr)))
               A_out_bw = pycu.view_as_complex(A.adjoint(pycu.view_as_real(A_out_fw)))
        """
        init_kwargs = _NUFFT3._sanitize_init_kwargs(
            x=x,
            z=z,
            isign=isign,
            eps=eps,
            real=real,
            **kwargs,
        )
        return _NUFFT3(**init_kwargs).squeeze()

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: pyct.NDArray (constructor-dependant)
            - **Type 1 and 3:**
                * (...,  M) input weights :math:`\mathbf{w} \in \mathbb{R}^{M}` (real transform).
                * (..., 2M) input weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.
                  (See :py:func:`~pycsou.util.complex.view_as_real`.)
            - **Type 2:**
                * (...,  N.prod()) input weights :math:`\mathbf{u} \in
                  \mathbb{R}^{\mathcal{I}_{N_1, \ldots, N_d}}` (real transform).
                * (..., 2N.prod()) input weights :math:`\mathbf{u} \in
                  \mathbb{C}^{\mathcal{I}_{N_1, \ldots, N_d}}` viewed as a real array.
                  (See :py:func:`~pycsou.util.complex.view_as_real`.)

        Returns
        -------
        out: pyct.NDArray (constructor-dependant)
            - **Type 1:**
                (..., 2N.prod()) output weights :math:`\mathbf{u} \in
                \mathbb{C}^{\mathcal{I}_{N_1, \ldots, N_d}}` viewed as a real array.
                (See :py:func:`~pycsou.util.complex.view_as_real`.)
            - **Type 2:**
                (..., 2M) output weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.
                (See :py:func:`~pycsou.util.complex.view_as_real`.)
            - **Type 3:**
                (..., 2N) output weights :math:`\mathbf{v} \in \mathbb{C}^{N}` viewed as a real array.
                (See :py:func:`~pycsou.util.complex.view_as_real`.)
        """
        raise NotImplementedError

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: pyct.NDArray (constructor-dependant)
            - **Type 1:**
                (..., 2N.prod()) output weights :math:`\mathbf{u} \in
                \mathbb{C}^{\mathcal{I}_{N_1, \ldots, N_d}}` viewed as a real array.
                (See :py:func:`~pycsou.util.complex.view_as_real`.)
            - **Type 2:**
                (..., 2M) output weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.
                (See :py:func:`~pycsou.util.complex.view_as_real`.)
            - **Type 3:**
                (..., 2N) output weights :math:`\mathbf{v} \in \mathbb{C}^{N}` viewed as a real array.
                (See :py:func:`~pycsou.util.complex.view_as_real`.)

        Returns
        -------
        out: pyct.NDArray (constructor-dependant)
            - **Type 1 and 3:**
                * (...,  M) input weights :math:`\mathbf{w} \in \mathbb{R}^{M}` (real transform).
                * (..., 2M) input weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.
                  (See :py:func:`~pycsou.util.complex.view_as_real`.)
            - **Type 2:**
                * (...,  N.prod()) input weights :math:`\mathbf{u} \in
                  \mathbb{R}^{\mathcal{I}_{N_1, \ldots, N_d}}` (real transform).
                * (..., 2N.prod()) input weights :math:`\mathbf{u} \in
                  \mathbb{C}^{\mathcal{I}_{N_1, \ldots, N_d}}` viewed as a real array.
                  (See :py:func:`~pycsou.util.complex.view_as_real`.)
        """
        raise NotImplementedError

    def lipschitz(self, **kwargs) -> pyct.Real:
        # Analytical form known if algo="fro"
        if kwargs.get("algo", "svds") == "fro":
            self._lipschitz = np.sqrt(self._M * np.prod(self._N))
        else:
            self._lipschitz = pyca.LinOp.lipschitz(self, **kwargs)
        return self._lipschitz

    def to_sciop(self, **kwargs):
        # _NUFFT.apply/adjoint() only support the precision provided at init-time.
        if not self._direct_eval:
            kwargs.update(dtype=self._x.dtype)  # silently drop user-provided `dtype`
        op = pyca.LinOp.to_sciop(self, **kwargs)
        return op

    @staticmethod
    def _as_canonical_coordinate(x: pyct.NDArray) -> pyct.NDArray:
        if (N_dim := x.ndim) == 1:
            x = x.reshape((-1, 1))
        elif N_dim == 2:
            assert 1 <= x.shape[-1] <= 3, "Only (1,2,3)-D transforms supported."
        else:
            raise ValueError(f"Expected 1D/2D array, got {N_dim}-D array.")
        return x

    @staticmethod
    def _as_canonical_mode(N) -> pyct.NDArrayShape:
        N = pycu.as_canonical_shape(N)
        assert all(_ > 0 for _ in N)
        assert 1 <= len(N) <= 3, "Only (1,2,3)-D transforms supported."
        return N

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        # check init() params + put in standardized form
        raise NotImplementedError

    @staticmethod
    def _plan_fw(**kwargs) -> finufft.Plan:
        # create plan and set points
        raise NotImplementedError

    def _fw(self, arr: pyct.NDArray) -> pyct.NDArray:
        # apply forward operator.
        # input: (n_trans, Q1) complex-valued
        # output: (n_trans, Q2) complex-valued
        raise NotImplementedError

    @staticmethod
    def _plan_bw(**kwargs) -> finufft.Plan:
        # create plan and set points
        raise NotImplementedError

    def _bw(self, arr: pyct.NDArray) -> pyct.NDArray:
        # apply backward operator.
        # input: (n_trans, Q2) complex-valued
        # output: (n_trans, Q1) complex-valued
        raise NotImplementedError

    @staticmethod
    def _preprocess(
        arr: pyct.NDArray,
        n_trans: pyct.Integer,
        dim_out: pyct.Integer,
    ):
        # Internal method for apply/adjoint.
        #
        # Parameters
        # ----------
        # arr: pyct.NDArray
        #     (..., N1) complex-valued input of [apply|adjoint]().
        # n_trans: pyct.Integer
        #     n_trans parameter given to finufft.Plan()
        # dim_out: pyct.Integer
        #     Trailing dimension [apply|adjoint](arr) should have.
        #
        # Returns
        # -------
        # x: pyct.NDArray
        #     (N_blk, n_trans, N1) complex-valued blocks to input to [_fw|_bw](), suitably augmented
        #     as needed.
        # N: pyct.Integer
        #     Amount of "valid" data to extract from [_fw|_bw](). {For _postprocess()}
        # sh_out: pyct.NDArrayShape
        #     Shape [apply|adjoint](arr) should have. {For _postprocess()}
        sh_out = arr.shape[:-1] + (dim_out,)
        if arr.ndim == 1:
            arr = arr.reshape((1, -1))
        N, dim_in = np.prod(arr.shape[:-1]), arr.shape[-1]

        N_blk, r = divmod(N, n_trans)
        N_blk += 1 if (r > 0) else 0
        if r == 0:
            x = arr
        else:
            xp = pycu.get_array_module(arr)
            x = xp.concatenate(
                [
                    arr.reshape((N, dim_in)),
                    xp.zeros((n_trans - r, dim_in), dtype=arr.dtype),
                ],
                axis=0,
            )
        x = x.reshape((N_blk, n_trans, dim_in))
        return x, N, sh_out

    @staticmethod
    def _scan(
        func: cabc.Callable[[pyct.NDArray], pyct.NDArray],
        data: list[pyct.NDArray],
        blk_shape: pyct.NDArrayShape = None,
    ) -> list[pyct.NDArray]:
        # Internal method for apply/adjoint.
        #
        # Computes the equivalent of ``map(func, data)``, with the constraint that `func` is applied
        # to each element of `data` in sequence.
        #
        # For Dask-array inputs, this amounts to creating a task graph with virtual dependencies
        # between successive `func` calls. In other words, the task graph looks like:
        #
        #        map(func, data)
        #
        #                    +----+              +----+
        #          data[0]-->|func|-->blk[0]-+-->|list|-->blks
        #             .      +----+          |   +----+
        #             .                      |
        #             .      +----+          |
        #          data[n]-->|func|-->blk[n]-+
        #                    +----+
        #
        # ==========================================================================================================
        #        _scan(func, data)
        #                                                                                             +----+
        #                              +-------------------+----------------+--------------------+----+list|-->blks
        #                              |                   |                |                    |    +----+
        #                              |                   |                |                    |
        #                    +----+    |        +----+     |      +---+     |         +----+     |
        #          data[0]-->|func|-->blk[0]-+->|func|-->blk[1]-->|...|-->blk[n-1]-+->|func|-->blk[n]
        #                    +----+          |  +----+            +---+            |  +----+
        #                                    |                                     |
        #          data[1]-------------------+                                     |
        #             .                                                            |
        #             .                                                            |
        #             .                                                            |
        #          data[n]---------------------------------------------------------+
        #
        #
        # Parameters
        # ----------
        # func: callable
        #     Function to apply to each element of `data`.
        # data: list[pyct.NDArray]
        #     (N_blk,) arrays of identical shape.
        # blk_shape: pyct.NDArrayShape
        #     Shape of ``func(data[i])``.
        #
        #     This hint is only required if inputs are dask arrays.
        #
        # Returns
        # -------
        # blks: list[pyct.NDArray]
        #     (N_blk,) arrays of shape `blk_shape`.
        NDI = pycd.NDArrayInfo
        if NDI.from_obj(data) == NDI.DASK:
            assert blk_shape is not None
            xp = NDI.DASK.module()

            blks = []
            for i in range(len(data)):
                _func = dgm.bind(
                    children=dask.delayed(func, pure=True),
                    parents=blks[i - 1] if (i > 0) else [],
                )
                blk = xp.from_delayed(
                    _func(data[i]),
                    shape=blk_shape,
                    dtype=data[i].dtype,
                )
                blks.append(blk)
        else:
            blks = [func(blk) for blk in data]
        return blks

    @staticmethod
    def _postprocess(
        blks: list[pyct.NDArray],
        N: pyct.Integer,
        sh_out: pyct.NDArrayShape,
    ) -> pyct.NDArray:
        # Internal method for apply/adjoint.
        #
        # Parameters
        # ----------
        # blks: list[pyct.NDArray]
        #     (N_blk,) complex-valued outputs of [_fw|_bw]().
        # N: pyct.Integer
        #     Amount of "valid" data to extract from [_fw|_bw]()
        # sh_out: pyct.NDArrayShape
        #     Shape [apply|adjoint](arr) should have.
        xp = pycu.get_array_module(blks[0])
        return xp.concatenate(blks, axis=0)[:N].reshape(sh_out)

    def ascomplexarray(
        self,
        xp: pyct.ArrayModule = np,
        dtype: pyct.DType = None,
    ) -> pyct.NDArray:
        r"""
        Matrix representation (complex-valued) of the linear operator.

        Parameters
        ----------
        xp: pyct.ArrayModule
            Which array module to use to represent the output.
        dtype: pyct.DType
            Optional (complex-valued) type of the array.

        Returns
        -------
        A: pyct.NDArray
            Array representation of the operator (NUFFT type-dependant).

            - **Type 1:** (N.prod(), M)
            - **Type 2:** (M, N.prod())
            - **Type 3:** (N, M)
        """
        raise NotImplementedError

    def mesh(
        self,
        xp: pyct.ArrayModule = np,
        dtype: pyct.DType = None,
        scale: str = "unit",
        upsampled: bool = False,
    ) -> pyct.NDArray:
        r"""
        For type1/2 NUFFT: compute the transform's meshgrid
        :math:`\mathcal{I}_{N_{1} \times \cdots \times N_{d}}
        =
        \mathcal{I}_{N_{1}} \times \cdots \times \mathcal{I}_{N_{d}}`.

        For type-3 NUFFT: compute the (shifted) meshgrid used for internal FFT computations.

        Parameters
        ----------
        xp: pyct.ArrayModule
            Which array module to use to represent the output.
        dtype: pyct.DType
            Optional type of the array.
        scale: str
            Grid scale. Options are:

            - **Type1 and 2:**
                * ``unit``, i.e. :math:`\mathcal{I} \in [[-N_{d}//2, \ldots, N_{d}//2 + 1]]^{d}`
                * ``source``, i.e. :math:`\mathcal{I} \in [-\pi, \pi)^{d}`
            - **Type 3:**
                * ``unit``, i.e. :math:`\mathcal{I} \in [[-N_{d}//2, \ldots, N_{d}//2 + 1]]^{d}`
                * ``source``, i.e. :math:`\mathcal{I}_{\text{source}} \in x_{d}^{c} + [-X_{d}, X_{d})^{d}`
                * ``target``, i.e. :math:`\mathcal{I}_{\text{target}} \in z_{d}^{c} + [-Z_{d}, Z_{d})^{d}`,
                  where :math:`x^{c}`, :math:`z^{c}` denote the data centroids, and :math:`X`,
                  :math:`Z` the data half-widths.
        upsampled: bool
            Use the upsampled meshgrid.
            (See [FINUFFT]_ for details.)

        Returns
        -------
        grid: pyct.NDArray
            (N1, ..., Nd, d) grid.

        Examples
        --------
        .. code-block:: python3

           import numpy as np
           import pycsou.operator.linop as pycl

           rng = np.random.default_rng(0)
           D, M, N = 1, 2, 3  # D denotes the dimension of the data
           x = np.fmod(rng.normal(size=(M, D)), 2 * np.pi)
           A = pycl.NUFFT.type1(
               x, N,
               isign=-1,
               eps=1e-3
           )
           A.mesh()  # array([[-1.],
                     #        [ 0.],
                     #        [ 1.]])
        """
        raise NotImplementedError

    def plot_kernel(self, ax=None, **kwargs):
        """
        Plot the spreading/interpolation kernel (along each dimension) on its support.

        Parameters
        ----------
        ax: :py:class:`~matplotlib.axes.Axes`
            Axes to draw on.
            If :py:obj:`None`, a new axes is used.
        **kwargs
            Keyword arguments forwarded to :py:meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        ax : :py:class:`~matplotlib.axes.Axes`

        Notes
        -----
        Requires `Matplotlib <https://matplotlib.org/>`_ to be installed.
        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError("This method requires matplotlib to be installed.")

        if ax is None:
            _, ax = plt.subplots()

        width = self._kernel_width()
        beta = self._kernel_beta()
        N = self._fft_shape()

        N_sample = 100
        z = np.linspace(-1, 1, N_sample)
        for d, n in zip(range(self._D), N):
            alpha = np.pi * width / n
            x = z / alpha
            phi = ES_kernel(x, beta)
            ax.plot(x, phi, label=rf"$\phi_{d}$", **kwargs)

        if self._D > 1:
            ax.legend()
        return ax

    def params(self) -> collections.namedtuple:
        r"""
        Compute internal parameters of the [FINUFFT]_ implementation.

        Returns
        -------
        p: namedtuple

        Internal parameters of the FINUFFT implementation, with fields:

        * upsample_factor: float
            FFT upsampling factor > 1
        * kernel_width: int
            Width of the spreading/interpolation kernels (in number of samples).
        * kernel_beta: float
            Kernel decay parameter :math:`\beta > 0`.
        * fft_shape: (d,) [int]
            Size of the D-dimensional FFT(s) performed internally.
        * dilation_factor: (d,) [float]
            Dilation factor(s) :math:`\gamma_{d}`. (Type-3 only)
        """
        if self._direct_eval:
            p = None
        else:
            FINUFFT_PARAMS = collections.namedtuple(
                "finufft_params",
                [
                    "upsample_factor",
                    "kernel_width",
                    "kernel_beta",
                    "fft_shape",
                    "dilation_factor",
                ],
            )
            p = FINUFFT_PARAMS(
                upsample_factor=self._upsample_factor(),
                kernel_width=self._kernel_width(),
                kernel_beta=self._kernel_beta(),
                fft_shape=self._fft_shape(),
                dilation_factor=self._dilation_factor(),
            )
        return p

    def _upsample_factor(self) -> pyct.Real:
        raise NotImplementedError

    def _kernel_width(self) -> pyct.Integer:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/spreadinterp.cpp::setup_spreader()
        # [FINUFFT]_
        #     eq   3.2
        #     sect 4.2
        u = self._upsample_factor()
        if np.isclose(u, 2):
            w = np.ceil(-np.log10(self._eps) + 1)
        else:  # 1.25
            scale = np.pi * np.sqrt(1 - (1 / u))
            w = np.ceil(-np.log(self._eps) / scale)
        w = max(2, int(w))
        return w

    def _kernel_beta(self) -> pyct.Real:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/spreadinterp.cpp::setup_spreader()
        # [FINUFFT]_
        #     eq   3.2
        #     sect 4.2
        u = self._upsample_factor()
        w = self._kernel_width()
        if np.isclose(u, 2):
            scale = {
                2: 2.20,
                3: 2.26,
                4: 2.38,
            }.get(w, 2.30)
        else:  # 1.25
            gamma = 0.97
            scale = gamma * np.pi * (1 - (0.5 / u))
        beta = float(scale * w)
        return beta

    def _fft_shape(self) -> pyct.NDArrayShape:
        raise NotImplementedError

    def _dilation_factor(self) -> cabc.Sequence[pyct.Integer]:
        raise NotImplementedError


class _NUFFT1(NUFFT):
    def __init__(self, **kwargs):
        self._M, self._D = kwargs["x"].shape  # Useful constants
        self._N = kwargs["N"]
        self._x = kwargs["x"]
        self._isign = kwargs["isign"]

        self._eps = kwargs.get("eps")
        self._direct_eval = not (self._eps > 0)
        self._real_in = kwargs.pop("real_in")
        self._real_out = kwargs.pop("real_out")
        self._upsampfac = kwargs.get("upsampfac")
        if self._direct_eval:
            self._plan = None
            self._n = kwargs.get("n_trans", 1)
        else:
            self._plan = dict(
                fw=self._plan_fw(**kwargs),
                bw=self._plan_bw(**kwargs),
            )
            self._n = self._plan["fw"].n_trans

        sh_op = [2 * np.prod(self._N), 2 * self._M]
        sh_op[0] //= 2 if self._real_out else 1
        sh_op[1] //= 2 if self._real_in else 1
        super().__init__(shape=sh_op)
        self._lipschitz = self.lipschitz(algo="fro")

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        kwargs = kwargs.copy()
        for k in ("nufft_type", "n_modes_or_dim", "dtype", "modeord"):
            kwargs.pop(k, None)
        x = kwargs["x"] = pycu.compute(cls._as_canonical_coordinate(kwargs["x"]))
        N = kwargs["N"] = cls._as_canonical_mode(kwargs["N"])
        kwargs["isign"] = int(np.sign(kwargs["isign"]))
        kwargs["eps"] = float(kwargs["eps"])
        kwargs["real_in"] = bool(kwargs["real_in"])
        kwargs["real_out"] = bool(kwargs["real_out"])
        if (D := x.shape[-1]) == len(N):
            pass
        elif len(N) == 1:
            kwargs["N"] = N * D
        else:
            raise ValueError("x vs. N: dimensionality mis-match.")
        return kwargs

    @staticmethod
    def _plan_fw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        x, N = [kwargs.pop(_) for _ in ("x", "N")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=1,
            n_modes_or_dim=N,
            dtype=pycrt.getPrecision().value,
            eps=kwargs.pop("eps"),
            n_trans=kwargs.pop("n_trans", 1),
            isign=kwargs.pop("isign"),
            modeord=0,
            **kwargs,
        )
        plan.setpts(**dict(zip("xyz"[:N_dim], x.T[:N_dim])))
        return plan

    def _fw(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._direct_eval:
            # Computing the target each time is wasteful (in comparison to the type-3 case where it
            # is implicit.) We are ok with this since relying on NUDFT is a failsafe.
            target = self.mesh(
                xp=pycd.NDArrayInfo.from_obj(arr).module(),
                dtype=self._x.dtype,
                scale="unit",
                upsampled=False,
            ).reshape((-1, self._D))

            out = _nudft(
                weight=arr,
                source=self._x,
                target=target,
                isign=self._isign,
                dtype=arr.dtype,
            )
        else:
            if self._n == 1:  # finufft limitation: insists on having no
                arr = arr[0]  # leading-dim if n_trans==1.
            out = self._plan["fw"].execute(arr)  # ([n_trans], M) -> ([n_trans], N1,..., Nd)
        return out.reshape((self._n, np.prod(self._N)))

    @staticmethod
    def _plan_bw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        x, N = [kwargs.pop(_) for _ in ("x", "N")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=2,
            n_modes_or_dim=N,
            dtype=pycrt.getPrecision().value,
            eps=kwargs.pop("eps"),
            n_trans=kwargs.pop("n_trans", 1),
            isign=-kwargs.pop("isign"),
            modeord=0,
            **kwargs,
        )
        plan.setpts(**dict(zip("xyz"[:N_dim], x.T[:N_dim])))
        return plan

    def _bw(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._direct_eval:
            # Computing the target each time is wasteful (in comparison to the type-3 case where it
            # is implicit.) We are ok with this since relying on NUDFT is a failsafe.
            target = self.mesh(
                xp=pycd.NDArrayInfo.from_obj(arr).module(),
                dtype=self._x.dtype,
                scale="unit",
                upsampled=False,
            ).reshape((-1, self._D))

            out = _nudft(
                weight=arr,
                source=target,
                target=self._x,
                isign=-self._isign,
                dtype=arr.dtype,
            )
        else:
            arr = arr.reshape((self._n, *self._N))
            if self._n == 1:  # finufft limitation: insists on having no
                arr = arr[0]  # leading-dim if n_trans==1.
            out = self._plan["bw"].execute(arr)  # ([n_trans], N1, ..., Nd) -> ([n_trans], M)
        return out.reshape((self._n, self._M))

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._real_in:
            r_width = pycrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pycu.view_as_complex(arr)

        data, N, sh = self._preprocess(arr, self._n, np.prod(self._N))
        blks = self._scan(self._fw, data, (self._n, np.prod(self._N)))
        out = self._postprocess(blks, N, sh)

        if self._real_out:
            return out.real
        else:
            return pycu.view_as_real(out)

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._real_out:
            r_width = pycrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pycu.view_as_complex(arr)

        data, N, sh = self._preprocess(arr, self._n, self._M)
        blks = self._scan(self._bw, data, (self._n, self._M))
        out = self._postprocess(blks, N, sh)

        if self._real_in:
            return out.real
        else:
            return pycu.view_as_real(out)

    def ascomplexarray(self, **kwargs) -> pyct.NDArray:
        # compute exact operator (using supported precision/backend)
        xp = pycu.get_array_module(self._x)
        mesh = self.mesh(
            xp=xp,
            dtype=self._x.dtype,
            scale="unit",
            upsampled=False,
        ).reshape((-1, self._D))
        _A = xp.exp(1j * self._isign * mesh @ self._x.T)

        # then comply with **kwargs()
        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        c_dtype = kwargs.get("dtype", pycrt.getPrecision().complex.value)
        A = xp.array(pycu.to_NUMPY(_A), dtype=c_dtype)
        return A

    def mesh(self, **kwargs) -> pyct.NDArray:
        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pycrt.getPrecision().value)
        scale = kwargs.get("scale", "unit")
        upsampled = kwargs.get("upsampled", False)

        N = self._fft_shape() if upsampled else self._N
        grid = xp.stack(  # (N1, ..., Nd, D)
            xp.meshgrid(
                *[xp.arange(-(n // 2), (n - 1) // 2 + 1, dtype=dtype) for n in N],
                indexing="ij",
            ),
            axis=-1,
        )
        if scale == "source":
            s = xp.array(2 * np.pi / np.array(N), dtype=dtype)
            grid *= s
        return grid

    def asarray(self, **kwargs) -> pyct.NDArray:
        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        if (r_dtype := kwargs.get("dtype")) is None:
            c_dtype = pycrt.getPrecision().complex.value
        else:
            r_width = pycrt.Width(r_dtype)
            c_dtype = r_width.complex.value
        _A = self.ascomplexarray(xp=xp, dtype=c_dtype)

        A = pycu.view_as_real_mat(
            cmat=_A,
            real_input=self._real_in,
            real_output=self._real_out,
        )
        return A

    def _upsample_factor(self) -> pyct.Real:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/finufft.cpp::FINUFFT_MAKEPLAN()
        if (u := self._upsampfac) is None:
            precQ = self._eps >= 1e-9
            dimQ = lambda d: self._D == d
            cutoffQ = lambda cutoff: np.prod(self._N) > int(cutoff)
            if precQ and dimQ(1) and cutoffQ(1e7):
                u = 1.25
            elif precQ and dimQ(2) and cutoffQ(3e5):
                u = 1.25
            elif precQ and dimQ(3) and cutoffQ(3e6):
                u = 1.25
            else:
                u = 2
        return u

    def _fft_shape(self) -> pyct.NDArrayShape:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/finufft.cpp::SET_NF_TYPE12()
        # [FINUFFT]_
        #     sect 3.1.1
        u = self._upsample_factor()
        w = self._kernel_width()
        shape = []
        for n in self._N:
            target = max(int(u * n), 2 * w)
            n_opt = pycu.next_fast_len(target, even=True)
            shape.append(n_opt)
        return tuple(shape)

    def _dilation_factor(self) -> cabc.Sequence[pyct.Integer]:
        # Undefined for type-1
        return None


class _NUFFT3(NUFFT):
    def __init__(self, **kwargs):
        self._M, self._D = kwargs["x"].shape  # Useful constants
        self._N, _ = kwargs["z"].shape
        self._x = kwargs["x"]
        self._z = kwargs["z"]
        self._isign = kwargs["isign"]

        self._eps = kwargs.get("eps")
        self._direct_eval = not (self._eps > 0)
        self._real = kwargs.pop("real")
        self._upsampfac = kwargs.get("upsampfac")
        if self._direct_eval:
            self._plan = None
            self._n = kwargs.get("n_trans", 1)
        else:
            self._plan = dict(
                fw=self._plan_fw(**kwargs),
                bw=self._plan_bw(**kwargs),
            )
            self._n = self._plan["fw"].n_trans

        sh_op = [2 * self._N, 2 * self._M]
        sh_op[1] //= 2 if self._real else 1
        super().__init__(shape=sh_op)
        self._lipschitz = self.lipschitz(algo="fro")

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        kwargs = kwargs.copy()
        for k in ("nufft_type", "n_modes_or_dim", "dtype", "modeord"):
            kwargs.pop(k, None)
        x = kwargs["x"] = pycu.compute(cls._as_canonical_coordinate(kwargs["x"]))
        z = kwargs["z"] = pycu.compute(cls._as_canonical_coordinate(kwargs["z"]))
        assert x.shape[-1] == z.shape[-1], "x vs. z: dimensionality mis-match."
        assert pycu.get_array_module(x) == pycu.get_array_module(z)
        kwargs["isign"] = int(np.sign(kwargs["isign"]))
        kwargs["eps"] = float(kwargs["eps"])
        kwargs["real"] = bool(kwargs["real"])
        return kwargs

    @staticmethod
    def _plan_fw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        x, z = [kwargs.pop(_) for _ in ("x", "z")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=3,
            n_modes_or_dim=N_dim,
            dtype=pycrt.getPrecision().value,
            eps=kwargs.pop("eps"),
            n_trans=kwargs.pop("n_trans", 1),
            isign=kwargs.pop("isign"),
            **kwargs,
        )
        plan.setpts(
            **dict(
                zip(
                    "xyz"[:N_dim] + "stu"[:N_dim],
                    (*x.T[:N_dim], *z.T[:N_dim]),
                )
            ),
        )
        return plan

    def _fw(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._direct_eval:
            out = _nudft(
                weight=arr,
                source=self._x,
                target=self._z,
                isign=self._isign,
                dtype=arr.dtype,
            )
        else:
            if self._n == 1:  # finufft limitation: insists on having no
                arr = arr[0]  # leading-dim if n_trans==1.
            out = self._plan["fw"].execute(arr)  # ([n_trans], M) -> ([n_trans], N)
        return out.reshape((self._n, self._N))

    @staticmethod
    def _plan_bw(**kwargs) -> finufft.Plan:
        kwargs = kwargs.copy()
        x, z = [kwargs.pop(_) for _ in ("x", "z")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=3,
            n_modes_or_dim=N_dim,
            dtype=pycrt.getPrecision().value,
            eps=kwargs.pop("eps"),
            n_trans=kwargs.pop("n_trans", 1),
            isign=-kwargs.pop("isign"),
            **kwargs,
        )
        plan.setpts(
            **dict(
                zip(
                    "xyz"[:N_dim] + "stu"[:N_dim],
                    (*z.T[:N_dim], *x.T[:N_dim]),
                )
            ),
        )
        return plan

    def _bw(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._direct_eval:
            out = _nudft(
                weight=arr,
                source=self._z,
                target=self._x,
                isign=-self._isign,
                dtype=arr.dtype,
            )
        else:
            if self._n == 1:  # finufft limitation: insists on having no
                arr = arr[0]  # leading-dim if n_trans==1.
            out = self._plan["bw"].execute(arr)  # ([n_trans], N) -> ([n_trans], M)
        return out.reshape((self._n, self._M))

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self._real:
            r_width = pycrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pycu.view_as_complex(arr)

        data, N, sh = self._preprocess(arr, self._n, self._N)
        blks = self._scan(self._fw, data, (self._n, self._N))
        out = self._postprocess(blks, N, sh)

        return pycu.view_as_real(out)

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        arr = pycu.view_as_complex(arr)

        data, N, sh = self._preprocess(arr, self._n, self._M)
        blks = self._scan(self._bw, data, (self._n, self._M))
        out = self._postprocess(blks, N, sh)

        if self._real:
            return out.real
        else:
            return pycu.view_as_real(out)

    def ascomplexarray(self, **kwargs) -> pyct.NDArray:
        # compute exact operator (using supported precision/backend)
        xp = pycu.get_array_module(self._x)
        _A = xp.exp(1j * self._isign * self._z @ self._x.T)

        # then comply with **kwargs()
        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        c_dtype = kwargs.get("dtype", pycrt.getPrecision().complex.value)
        A = xp.array(pycu.to_NUMPY(_A), dtype=c_dtype)
        return A

    def mesh(self, **kwargs) -> pyct.NDArray:
        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pycrt.getPrecision().value)
        scale = kwargs.get("scale", "unit")
        upsampled = True or kwargs.pop("upsampled", True)  # upsampled unsupported for type-3
        kwargs = dict(
            xp=xp,
            dtype=dtype,
            upsampled=upsampled,
        )

        if scale == "unit":
            grid = _NUFFT1.mesh(self, scale="unit", **kwargs)
        else:
            grid = _NUFFT1.mesh(self, scale="source", **kwargs)
            f = lambda _: xp.array(_, dtype=dtype)
            if scale == "source":
                s = f(self._dilation_factor())
                grid *= s
                _, center = self.__shift_coords(self._x)
            else:  # target
                s = f(self._dilation_factor()) / f(self._fft_shape())
                s *= f(2 * np.pi * self._upsample_factor())
                grid /= s
                _, center = self.__shift_coords(self._z)
            grid += f(center)
        return grid

    def asarray(self, **kwargs) -> pyct.NDArray:
        xp = kwargs.get("xp", pycd.NDArrayInfo.NUMPY.module())
        if (r_dtype := kwargs.get("dtype")) is None:
            c_dtype = pycrt.getPrecision().complex.value
        else:
            r_width = pycrt.Width(r_dtype)
            c_dtype = r_width.complex.value
        _A = self.ascomplexarray(xp=xp, dtype=c_dtype)

        A = pycu.view_as_real_mat(cmat=_A, real_input=self._real)
        return A

    def _upsample_factor(self) -> pyct.Real:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/finufft.cpp::FINUFFT_MAKEPLAN()
        if (u := self._upsampfac) is None:
            if self._eps >= 1e-9:
                u = 1.25
            else:
                u = 2
        return u

    def _fft_shape(self) -> pyct.NDArrayShape:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/finufft.cpp::set_nhg_type3()
        # [FINUFFT]_
        #     eq 3.23
        u = self._upsample_factor()
        w = self._kernel_width()
        X, _ = self.__shift_coords(self._x)  # (D,)
        S, _ = self.__shift_coords(self._z)  # (D,)
        shape = []
        for d in range(self._D):
            n = (2 * u * max(1, X[d] * S[d]) / np.pi) + (w + 1)
            target = max(int(n), 2 * w)
            n_opt = pycu.next_fast_len(target, even=True)
            shape.append(n_opt)
        return tuple(shape)

    def _dilation_factor(self) -> cabc.Sequence[pyct.Integer]:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/finufft.cpp::set_nhg_type3()
        # [FINUFFT]_
        #     eq 3.23
        u = self._upsample_factor()
        N = self._fft_shape()
        S, _ = self.__shift_coords(self._z)  # (D,)
        gamma = [n / (2 * u * s) for (n, s) in zip(N, S)]
        return tuple(gamma)

    @staticmethod
    def __shift_coords(pts: pyct.NDArray) -> pyct.NDArray:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/utils.cpp::arraywidcen()
        #     ./include/finufft/defs.h
        #
        # Parameters
        # ----------
        # pts: pyct.NDArray
        #     (Q, D) coordinates.
        #
        # Returns
        # -------
        # h_width: np.ndarray
        #     (D,) shifted half_width
        # center: np.ndarray
        #     (D,) shifted centroid
        low = pycu.to_NUMPY(pts.min(axis=0))
        high = pycu.to_NUMPY(pts.max(axis=0))
        h_width = (high - low) / 2
        center = (high + low) / 2
        grow_frac = 0.1

        mask = np.fabs(center) < h_width * grow_frac
        h_width[mask] += np.fabs(center[mask])
        center[mask] = 0
        return h_width, center


@numba.njit(parallel=True, fastmath=True, nogil=True)
def _nudft_NUMPY(
    weight: pyct.NDArray,  # (Q, M) weights (n_trans=Q) [complex64/128]
    source: pyct.NDArray,  # (M, D) sample points [float32/64]
    target: pyct.NDArray,  # (N, D) query points [float32/64]
    *,
    isign: SignT,
    dtype: pyct.DType,  # complex64/128
) -> pyct.NDArray:  # (Q, N) complex64/128
    Q = weight.shape[0]
    M = source.shape[0]
    N = target.shape[0]
    out = np.zeros(shape=(Q, N), dtype=dtype)
    for n in numba.prange(N):
        for m in range(M):
            scale = np.exp(isign * 1j * np.dot(source[m, :], target[n, :]))
            out[:, n] += weight[:, m] * scale
    return out


def _nudft_CUPY(
    weight: pyct.NDArray,  # (Q, M) weights (n_trans=Q) [complex64/128]
    source: pyct.NDArray,  # (M, D) sample points [float32/64]
    target: pyct.NDArray,  # (N, D) query points [float32/64]
    *,
    isign: SignT,
    dtype: pyct.DType,  # complex64/128
) -> pyct.NDArray:  # (Q, N) complex64/128
    @numba.cuda.jit(device=True)
    def _cexp(s, a, b):  # [(1,), (D,), (D,)] -> (1,)
        # np.exp(1j * s * (a @ b))
        D, c = len(a), 0
        for d in range(D):
            c += a[d] * b[d]
        out = cmath.exp(1j * s * c)
        return out

    @numba.cuda.jit(fastmath=True, opt=True, cache=True)
    def _kernel(weight, source, target, isign, out):
        Q, M = weight.shape[:2]
        N, D = target.shape[:2]
        q, n = numba.cuda.grid(2)
        if (q < Q) and (n < N):
            for m in range(M):
                scale = _cexp(isign, source[m, :], target[n, :])
                out[q, n] += weight[q, m] * scale

    Q = weight.shape[0]
    N = target.shape[0]
    xp = pycu.get_array_module(weight)
    out = xp.zeros((Q, N), dtype=dtype)

    ceil = lambda _: int(np.ceil(_))
    t_max = weight.device.attributes["MaxThreadsPerBlock"]
    tpb = [min(Q, t_max // 2), None]  # thread_per_block
    tpb[1] = t_max // tpb[0]
    bpg = [ceil(Q / tpb[0]), ceil(N / tpb[1])]  # block_per_grid

    config = _kernel[tuple(bpg), tuple(tpb)]
    config(weight, source, target, isign, out)
    return out


@pycu.redirect(i="weight", NUMPY=_nudft_NUMPY, CUPY=_nudft_CUPY)
def _nudft(
    weight: pyct.NDArray,  # (Q, M) weights (n_trans=Q) [complex64/128]
    source: pyct.NDArray,  # (M, D) sample points [float32/64]
    target: pyct.NDArray,  # (N, D) query points [float32/64]
    *,
    isign: SignT,
    dtype: pyct.DType,  # complex64/128
) -> pyct.NDArray:  # (Q, N) complex64/128
    pass
