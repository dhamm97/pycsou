import collections.abc as cabc
import types
import typing as typ

import dask.array as da
import dask.distributed as dad
import finufft
import numba as nb
import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "NUFFT",
]

SignT = typ.Literal[1, -1]
eps_default = 1e-4


class NUFFT(pyca.LinOp):
    r"""
    Non-Uniform Fast Fourier Transform (NUFFT) of Type 1/2/3 (for :math:`d=\{1,2,3\}`).

    The *Non-Uniform Fast Fourier Transform (NUFFT)* generalizes the FFT to off-grid data.
    There are three main types of NUFFTs proposed in the literature:

    * Type 1 (*non-uniform to uniform*),
    * Type 2 (*uniform to non-uniform*),
    * Type 3 (*non-uniform to non-uniform*).

    See the notes below as well as [FINUFFT]_ for definitions of the various transform types and
    algorithmic details.

    The transforms should be instantiated via
    :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type1`,
    :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type2`, and
    :py:meth:`~pycsou.operator.linop.nufft.NUFFT.type3` respectively (see each method for usage examples).

    The dimension of the NUFFT transforms is inferred from the dimensions of the input arguments,
    with support for for :math:`d=\{1,2,3\}`.

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
    :math:`\varepsilon>0`, the following exponential sums:

    .. math::

       &u_{\mathbf{n}} = \sum_{j=1}^{M} w_{j} e^{\sigma i\langle \mathbf{n}, \mathbf{x}_{j} \rangle}, \quad &\mathbf{n}\in \mathcal{I}_{N_1,\ldots, N_d},\qquad &\text{Type 1 (non-uniform to uniform)}\\
       &w_{j} = \sum_{\mathbf{n}\in\mathcal{I}_{N_1,\ldots, N_d}} u_{\mathbf{n}} e^{\sigma i\langle \mathbf{n}, \mathbf{x}_{j} \rangle }, \quad &j=1,\ldots, M,\qquad  &\text{Type 2 (uniform to non-uniform)}\\
       &v_{k} = \sum_{j=1}^{M} w_{j} e^{\sigma i\langle \mathbf{z}_k, \mathbf{x}_{j} \rangle }, \quad &k=1,\ldots, N, \qquad &\text{Type 3 (non-uniform to non-uniform)}


    where :math:`\sigma \in \{+1, -1\}` defines the sign of the transforms and
    :math:`u_{\mathbf{n}}, v_{k}, w_{j}\in \mathbb{C}`.
    For the type-1 and type-2 NUFFTs, the non-uniform samples :math:`\mathbf{x}_{j}` are assumed to
    lie in :math:`[-\pi,\pi)^d`.
    For the type-3 NUFFT, the non-uniform samples :math:`\mathbf{x}_{j}` and
    :math:`\mathbf{z}_{k}` are arbitrary points in :math:`\mathbb{R}^d`.

    **Adjoint NUFFTs.**
    The type-1 and type-2 NUFFTs with opposite signs form an *adjoint pair*.
    The adjoint of the type-3 NUFFT is obtained by flipping the transform's sign and switching the
    roles of :math:`\mathbf{z}_k` and :math:`\mathbf{x}_{j}` in (3).

    **Lipschitz Constants.**
    The type-1 NUFFT can be interpreted as the truncated Fourier Series of a :math:`2\pi`-periodic
    Dirac stream with innovations :math:`(w_j, \mathbf{x}_j)`.

    From Parseval's equality, we have hence

    .. math::

       \|u_{\mathbf{n}}\|^2
       =
       \frac{1}{2\pi} \left\|\sum_{j=1}^M w_j d_{N_1,\ldots,N_d}(\cdot-\mathbf{x}_j)\right\|_2^2
       =
       \frac{1}{2\pi}\mathbf{w}^HG\mathbf{w}
       \leq
       \frac{1}{2\pi} \|G\|_2\|\mathbf{w}\|_2^2,

    where :math:`d_{N_1,\ldots,N_d}:[-\pi, \pi)^d \to \mathbb{R}` is the :math:`d`-dimensional
    Dirichlet kernel of bandwidth :math:`(N_1,\ldots,N_d)` and :math:`G\in\mathbb{R}^{M \times M}`
    is the Gram matrix with entries :math:`G_{ij}=d_{N_1,\ldots,N_d}(\mathbf{x}_i-\mathbf{x}_j)`.
    The Lipschitz constant of the type-1 NUFFT is then proportional to the square root of the
    largest singular value of :math:`G`.
    Since the Gram is positive semi-definite, its largest eigenvalue can be bounded by its trace,
    which yields :math:`L = \sqrt{\|G\|_2/2\pi}\leq \sqrt{M\Pi_{i=1}^d N_i/2\pi}`.
    For the type-3 NUFFT, we bound the Lipschitz constant by the Frobenius norm of the operator,
    which yields :math:`L \leq \sqrt{NM}`.
    Note that these Lipschitz constants are cheap to compute but can be quite pessimistic. Tighter
    Lipschitz constants can be computed by calling the method
    :py:meth:`~pycsou.abc.operator.LinOp.lipschitz`.

    **Error Analysis.**
    Let :math:`\tilde{\mathbf{u}}\in\mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` and
    :math:`\tilde{\mathbf{w}}\in\mathbb{C}^M` be the outputs of the type-1 and type-2 NUFFT
    algorithms which approximate the sequences
    :math:`{\mathbf{u}}\in\mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` and
    :math:`{\mathbf{w}}\in\mathbb{C}^M` defined in (1) and (2) respectively.
    Then [FINUFFT]_ shows that the relative errors
    :math:`\|\tilde{\mathbf{u}}-{\mathbf{u}}\|_2/\|{\mathbf{u}}\|_2` and
    :math:`\|\tilde{\mathbf{w}}-{\mathbf{w}}\|_2/\|{\mathbf{w}}\|_2` are **almost always similar to
    the user-requested tolerance** :math:`\varepsilon`, except when round-off error dominates
    (i.e. very small user-requested tolerances).
    The same holds approximately for the type-3 NUFFT.
    Note however that this is a *typical error analysis*: some degenerate (but rare) worst-case
    scenarios can result in much higher errors.


    **Complexity.**
    Naive evaluation of the exponential sums (1), (2) and (3) above costs :math:`O(NM)`, where
    :math:`N=N_1\ldots N_d` for the type-1 and type-2 NUFFTs.
    NUFFT algorithms approximate these sums to a user-specified relative tolerance
    :math:`\varepsilon` in log-linear complexity in both :math:`N` and :math:`M`.
    More specifically, the complexity of the various NUFFTs are given by (see [FINUFFT]_):

    .. math::

       &\mathcal{O}\left(N\log(N) + M|\log(\varepsilon)|^d\right)\qquad &\text{(Types 1 and 2)}\\
       &\mathcal{O}\left(\Pi_{i=1}^dX_iZ_i\sum_{i=1}^d\log(X_iZ_i) + (M + N)|\log(\varepsilon)|^d\right)\qquad &\text{(Type 3)}

    where :math:`X_i = \max_{j=1,\ldots,M}|(\mathbf{x}_j)_i|` and :math:`Z_i =
    \max_{k=1,\ldots,N}|(\mathbf{z}_k)_i|` for :math:`i=1,\ldots,d`.
    The two terms intervening in the complexities above correspond to the complexity of the FFT and
    spreading/interpolation steps respectively.

    The complexity of the type-3 NUFFT can be arbitrarily large for poorly-centered data. In certain
    cases however, an easy fix consists in centering the data before/after the NUFFT via
    pre/post-phasing operations, as described in equation (3.24) of [FINUFFT]_.
    This optimization is automatically carried out by FINUFFT if the compute/memory gains are significant
    enough. [#]_

    **Backend.** The NUFFT tansforms are computed via Python wrappers to `FINUFFT
    <https://github.com/flatironinstitute/finufft>`_ and `cuFINUFFT
    <https://github.com/flatironinstitute/cufinufft>`_ (see also [FINUFFT]_ and [cuFINUFFT]_).
    These librairies perform the expensive spreading/interpolation between nonuniform points and the
    fine grid via the "exponential of semicircle" kernel (see [FINUFFT]_).

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
        This parameter is ignored by :py:class:`~pycsou.operator.linop.nufft.NUFFT`.
        Use the context manager :py:class:`~pycsou.runtime.Precision` to control the floating point
        precision.

    .. warning::

        The NUFFT is performed in **chunks of size (n_trans, K)** where K is the size of the non-stacking dimension of input arrays and
        n_trans the number of simultaneous transforms requested (see the ``n_trans`` parameter of `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.).
        To avoid memory overloading and bad performances, one must hence make sure that *each chunk fits comfortably in memory*.
        Note that this recommendation applies to Dask arrays too, which are rechunked to meet such chunk sizes (FINUFFT cannot process arbitrary chunk sizes).

        Additionally, this class is **only compatible** with Dask's `distributed scheduler <https://distributed.dask.org/en/stable/>`_ in multithreading mode (``processes=False``). [#]_
        Indeed, the sharing of FINUFFT's plan among all workers creates a race condition which, unlike Dask's default schedulers, the distributed scheduler can prevent by means of locks.
        One implication of this implementation is that chunks are processed serially one after the other, while NUFFT computations within each chunk are multithreaded.
        Note moreover that **multiprocessing is currently not available** due to serialization issues of FINUFFT's C routines (will be fixed in the future).

    .. [#] FINUFFT uses the following rule of thumb:
           for a given dimension, if the magnitude of the center is less than 10% of half the
           peak-to-peak distance, then the data is considered well-centered and no fix is performed.
    .. [#] The distributed scheduler in multithreading mode can be invoked as ``from dask.distributed import Client; client=Client(processes=False)``.


    See Also
    --------
    FFT, DCT, Radon
    """

    # The goal of this wrapper class is to sanitize __init__() inputs.

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)

    @staticmethod
    @pycrt.enforce_precision(i="x", o=False, allow_None=False)
    def type1(
        x: pyct.NDArray,
        N: typ.Union[pyct.Integer, tuple[pyct.Integer, ...]],
        isign: SignT = 1,
        eps: pyct.Real = eps_default,
        real: bool = False,
        **kwargs,
    ) -> pyca.LinOp:
        r"""
        Type 1 NUFFT (non-uniform to uniform).

        Parameters
        ----------
        x: NDArray
            (M, [d]) d-dimensional sample points :math:`\mathbf{x}_{j} \in \mathbb{R}^{d}`.
        N: int | tuple[int]
            ([d],) mesh size in each dimension :math:`(N_1, \ldots, N_d)`.
            If `N` is an integer, then the mesh is assumed to have the same size in each dimension.
        isign: 1 | -1
            Sign :math:`\sigma` of the transform.
        eps: float
            Requested relative accuracy (defaults to 1e-4).
        real: bool
            If ``True``, assumes ``.apply()`` takes (..., M) inputs.
            If ``False``, then ``.apply()`` takes (..., 2M) inputs.
        **kwargs
            Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
            (Illegal keywords are dropped silently.)
            Most useful are ``n_trans``, ``nthreads`` and ``debug`` (for debugging or diagnostics).

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
                       x,
                       N,
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
            real_input=real,
            real_output=False,
            **kwargs,
        )
        return _NUFFT1(**init_kwargs).squeeze()

    @staticmethod
    @pycrt.enforce_precision(i="x", o=False, allow_None=False)
    def type2(
        x: pyct.NDArray,
        N: typ.Union[pyct.Integer, tuple[pyct.Integer, ...]],
        isign: SignT = 1,
        eps: pyct.Real = eps_default,
        real: bool = False,
        **kwargs,
    ) -> pyca.LinOp:
        r"""
        Type 2 NUFFT (uniform to non-uniform).

        Parameters
        ----------
        x: NDArray
            (M, [d]) d-dimensional query points :math:`\mathbf{x}_{j} \in \mathbb{R}^{d}`.
        N: int | tuple[int]
            ([d],) mesh size in each dimension :math:`(N_1, \ldots, N_d)`.
            If `N` is an integer, then the mesh is assumed to have the same size in each dimension.
        isign: 1 | -1
            Sign :math:`\sigma` of the transform.
        eps: float
            Requested relative accuracy (defaults to 1e-4).
        real: bool
            If ``True``, assumes ``.apply()`` takes (..., N.prod()) inputs.
            If ``False``, then ``.apply()`` takes (..., 2N.prod()) inputs.
        **kwargs
            Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
            (Illegal keywords are dropped silently.)
            Most useful are ``n_trans``, ``nthreads`` and ``debug`` (for debugging or diagnostics).

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
                       x,
                       N,
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
            real_input=False,
            real_output=real,
            **kwargs,
        )

        op = _NUFFT1(**init_kwargs)

        def adj_complex_matrix(_, xp=np):
            return op.complex_matrix(xp).conj().T

        adj = op.squeeze().T
        adj.complex_matrix = types.MethodType(adj_complex_matrix, adj)
        return adj

    @staticmethod
    @pycrt.enforce_precision(i=("x", "z"), o=False, allow_None=False)
    def type3(
        x: pyct.NDArray,
        z: pyct.NDArray,
        isign: SignT = 1,
        eps: pyct.Real = eps_default,
        real: bool = False,
        **kwargs,
    ) -> pyca.LinOp:
        r"""
        Type 3 NUFFT (non-uniform to non-uniform).

        Parameters
        ----------
        x: NDArray
            (M, [d]) d-dimensional sample points :math:`\mathbf{x}_{j} \in \mathbb{R}^{d}`.
        z: NDArray
            (N, [d]) d-dimensional query points :math:`\mathbf{z}_{k} \in \mathbb{R}^{d}`.
        isign: 1 | -1
            Sign :math:`\sigma` of the transform.
        eps: float
            Requested relative accuracy (defaults to 1e-4).
        real: bool
            If ``True``, assumes ``.apply()`` takes (..., M) inputs.
            If ``False``, then ``.apply()`` takes (..., 2M) inputs.
        **kwargs
            Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
            (Illegal keywords are dropped silently.)
            Most useful are ``n_trans``, ``nthreads`` and ``debug`` (for debugging or diagnostics).

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
                  (see :py:func:`~pycsou.util.complex.view_as_real`).
            - **Type 2:**
                * (...,  N.prod()) output weights :math:`\mathbf{u} \in
                  \mathbb{R}^{\mathcal{I}_{N_1,\ldots, N_d}}` (real transform).
                * (..., 2N.prod()) output weights :math:`\mathbf{u} \in
                  \mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` viewed as a real array.
                  (see :py:func:`~pycsou.util.complex.view_as_real`).

        Returns
        -------
        out: pyct.NDArray (constructor-dependant)
            - **Type 1:**
                (..., 2N.prod()) output weights :math:`\mathbf{u} \in
                \mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` viewed as a real array.
                (see :py:func:`~pycsou.util.complex.view_as_real`).
            - **Type 2:**
                (..., 2M) input weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.
                (see :py:func:`~pycsou.util.complex.view_as_real`).
            - **Type 3:**
                (..., 2N) output weights :math:`\mathbf{v} \in \mathbb{C}^{N}` viewed as a real array.
                (see :py:func:`~pycsou.util.complex.view_as_real`).
        """
        raise NotImplementedError

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: pyct.NDArray (constructor-dependant)
            - **Type 1:**
                (..., 2N.prod()) output weights :math:`\mathbf{u} \in
                \mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` viewed as a real array.
                (see :py:func:`~pycsou.util.complex.view_as_real`).
            - **Type 2:**
                (..., 2M) input weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.
                (see :py:func:`~pycsou.util.complex.view_as_real`).
            - **Type 3:**
                (..., 2N) output weights :math:`\mathbf{v} \in \mathbb{C}^{N}` viewed as a real array.
                (see :py:func:`~pycsou.util.complex.view_as_real`).

        Returns
        -------
        out: pyct.NDArray (constructor-dependant)
            - **Type 1 and 3:**
                * (...,  M) input weights :math:`\mathbf{w} \in \mathbb{R}^{M}` (real transform).
                * (..., 2M) input weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.
                  (see :py:func:`~pycsou.util.complex.view_as_real`).
            - **Type 2:**
                * (...,  N.prod()) output weights :math:`\mathbf{u} \in
                  \mathbb{R}^{\mathcal{I}_{N_1,\ldots, N_d}}` (real transform).
                * (..., 2N.prod()) output weights :math:`\mathbf{u} \in
                  \mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` viewed as a real array.
                  (see :py:func:`~pycsou.util.complex.view_as_real`).
        """
        raise NotImplementedError

    def complex_matrix(self, xp: pyct.ArrayModule = np) -> pyct.NDArray:
        r"""
        Form the complex-valued matrix associated to the operator.

        Parameters
        ----------
        xp: ArrayModule
            Which array module to use to represent the output.

        Returns
        -------
        NDArray
            * (N.prod(), M) complex-valued array for type-1 NUFFT.
            * (M, N.prod()) complex-valued array for type-2 NUFFT.
            * (M, N) complex-valued array for type-3 NUFFT.
        Examples
        --------
        .. code-block:: python3

           import numpy as np
           import pycsou.operator.linop as pycl
           import pycsou.runtime as pycrt
           import pycsou.util as pycu

           rng = np.random.default_rng(0)
           D, M, N = 1, 2, 3  # D denotes the dimension of the data
           x = np.fmod(rng.normal(size=(M, D)), 2 * np.pi)
           A = pycl.NUFFT.type1(
                   x,
                   N,
                   isign=-1,
                   eps=1e-3,
               )
           A.complex_matrix()
           >> array([[0.99210636+0.12539922j, 0.99128684-0.13172096j],
                     [1.        +0.j        , 1.        +0.j        ],
                     [0.99210636-0.12539922j, 0.99128684+0.13172096j]])

        Warnings
        --------
        This method is mainly for debugging/benchmarking purposes and must be used sparingly/carefully as forming the complex
        matrix associated to the NUFFT operator can be slow and very memory intensive in high dimensional settings.
        """
        raise NotImplementedError

    def mesh(self, xp: pyct.ArrayModule = np):
        r"""
        For type-1 and type-2 NUFFTs, form the transform's meshgrid :math:`\mathcal{I}_{N_1\times\cdots\times N_d} =\mathcal{I}_{N_1}\times \cdots \times \mathcal{I}_{N_d}  \subset \mathbb{Z}^d`.

        Parameters
        ----------
        xp: ArrayModule
            Which array module to use to represent the output.

        Returns
        -------
        NDArray
            (N1, ..., Nd, d) meshgrid following Numpy's :py:func:`~numpy.meshgrid` convention with ``'ij'`` indexing.
            If ``modeord=1`` was passed as optional keyword argument to the class constructor, all axes of the grid are "`ifftshifted <https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html>`_"
            to reflect the different ordering convention.

        Examples
        --------
        .. code-block:: python3

           import numpy as np
           import pycsou.operator.linop as pycl

           rng = np.random.default_rng(0)
           D, M, N = 1, 2, 3  # D denotes the dimension of the data
           x = np.fmod(rng.normal(size=(M, D)), 2 * np.pi)
           A = pycl.NUFFT.type1(
                   x,
                   N,
                   isign=-1,
                   eps=1e-3,
               )
           A.mesh()
           >> array([[-1.,  0.,  1.]])
        """
        raise NotImplementedError

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
    def _as_canonical_mode(N) -> tuple[pyct.Integer]:
        if not isinstance(N, cabc.Sequence):
            N = (N,)
        N = tuple(map(int, N))
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
        # sh_out: tuple[pyct.Integer]
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
    def _postprocess(
        blks: list[pyct.NDArray],
        N: pyct.Integer,
        sh_out: tuple[pyct.Integer],
    ) -> pyct.NDArray:
        # Internal method for apply/adjoint.
        #
        # Parameters
        # ----------
        # blks: list[NDArray]
        #     (N_blk,) complex-valued outputs of [_fw|_bw]().
        # N: pyct.Integer
        #     Amount of "valid" data to extract from [_fw|_bw]()
        # sh_out: tuple[pyct.Integer]
        #     Shape [apply|adjoint](arr) should have.
        xp = pycu.get_array_module(blks[0])
        return xp.concatenate(blks, axis=0)[:N].reshape(sh_out)


class _NUFFT1(NUFFT):
    def __init__(self, **kwargs):
        self._real_input = kwargs.pop("real_input")
        self._real_output = kwargs.pop("real_output")
        self._eps = kwargs["eps"]
        if self._eps > 0:
            self._plan = dict(
                fw=self._plan_fw(**kwargs),
                bw=self._plan_bw(**kwargs),
            )
            self._n = self._plan["fw"].n_trans
        else:
            self._plan = None
        self._M, self._D = kwargs["x"].shape  # Useful constants
        self._N = kwargs["N"]
        self._x = kwargs["x"]
        self._isign = kwargs["isign"]
        self._modeord = kwargs.get("modeord", 0)

        sh_op = [2 * np.prod(self._N), 2 * self._M]
        if self._real_output:
            sh_op[0] //= 2
        if self._real_input:
            sh_op[1] //= 2
        super().__init__(shape=sh_op)
        self._lipschitz = np.sqrt(self._M * np.prod(self._N) / 2 * np.pi)

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        kwargs = kwargs.copy()
        for k in ("nufft_type", "n_modes_or_dim", "dtype"):
            kwargs.pop(k, None)
        x = kwargs["x"] = cls._as_canonical_coordinate(kwargs["x"])
        N = kwargs["N"] = cls._as_canonical_mode(kwargs["N"])
        kwargs["isign"] = int(np.sign(kwargs["isign"]))
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
            modeord=kwargs.pop("modeord", 0),
            **kwargs,
        )
        plan.setpts(
            **dict(zip("xyz"[:N_dim], pycu.compute(x.T[:N_dim]).astype(pycrt.getPrecision().value)))
        )  # astype() is needed here because dask.distributed passes a dtype that FINUFFT does not recognize as the builtin np.dtype('float64') (== passes but not "is" which FINUFFT uses)
        return plan

    def _fw(self, arr: np.ndarray) -> np.ndarray:
        out = self._plan["fw"].execute(arr.squeeze())  # ([n_trans], M) -> ([n_trans], N1,..., Nd)
        return out.reshape((self._n, np.prod(self._N)))

    def _fw_locked(self, arr: np.ndarray, lock: dad.Lock = None) -> np.ndarray:
        arr = arr.astype(
            pycrt.getPrecision().complex.value
        )  # astype() is needed here because dask.distributed passes a dtype that FINUFFT does not recognize as the builtin np.dtype('complex128') (== passes but not "is" which FINUFFT uses)
        with lock:
            out = self._plan["fw"].execute(arr.squeeze())  # ([n_trans], M) -> ([n_trans], N1,..., Nd)
        return out.reshape((self._n, np.prod(self._N)))[
            None, ...
        ]  # with map_blocks we can skip _postprocess by simply recreating the stacking axis after processing each chunk

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
            modeord=kwargs.pop("modeord", 0),
            **kwargs,
        )
        plan.setpts(
            **dict(zip("xyz"[:N_dim], pycu.compute(x.T[:N_dim]).astype(pycrt.getPrecision().value)))
        )  # astype() is needed here because dask.distributed passes a dtype that FINUFFT does not recognize as the builtin np.dtype('float64') (== passes but not "is" which FINUFFT uses)
        return plan

    def _bw(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.reshape((self._n, *self._N)).squeeze()
        out = self._plan["bw"].execute(arr)  # ([n_trans], N1, ..., Nd) -> ([n_trans], M)
        return out.reshape((self._n, self._M))  # req. if squeeze-like behaviour above kicked in.

    def _bw_locked(self, arr: np.ndarray, lock: dad.Lock = None) -> np.ndarray:
        arr = arr.reshape((self._n, *self._N)).squeeze()
        arr = arr.astype(
            pycrt.getPrecision().complex.value
        )  # astype() is needed here because dask.distributed passes a dtype that FINUFFT does not recognize as the builtin np.dtype('complex128') (== passes but not "is" which FINUFFT uses)
        with lock:
            out = self._plan["bw"].execute(arr)  # ([n_trans], N1, ..., Nd) -> ([n_trans], M)
        return out.reshape((self._n, self._M))[
            None, ...
        ]  # with map_blocks we can skip _postprocess by simply recreating the stacking axis after processing each chunk

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: pyct.NDArray (constructor-dependant)
            (...,  M) input weights :math:`\mathbf{w} \in \mathbb{R}^{M}`.
            (..., 2M) input weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.
            (see :py:func:`~pycsou.util.complex.view_as_real`.)

        Returns
        -------
        out: pyct.NDArray (constructor-dependant)
            (...,  N.prod()) output weights :math:`\mathbf{u} \in
            \mathbb{R}^{\mathcal{I}_{N_1,\ldots, N_d}}`
            (..., 2N.prod()) output weights :math:`\mathbf{u} \in
            \mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` viewed as a real array.
            (see :py:func:`~pycsou.util.complex.view_as_real`.)
        """
        if self._real_input:
            r_width = pycrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pycu.view_as_complex(arr)

        if self._eps > 0:
            out = self._nufft_apply(arr)
        else:
            out = self._nudft_apply(arr)
        return out.real if self._real_output else pycu.view_as_real(out)

    def _nufft_apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        data, N, sh = self._preprocess(arr, self._n, np.prod(self._N))
        if isinstance(data, da.Array):
            try:
                lock = dad.Lock()  # Use lock to avoid race condition because of the shared FFTW resources.
            except:
                error_message = r"""
                        The NUFFT operator requires Dask's distributed scheduler in multi-threading mode (other schedulers are not accepted).
                        Start a client and point it to the scheduler address:
                            from dask.distributed import Client
                            client = Client('ip-addr-of-scheduler:8786', processes=False)
                        """
                raise ValueError(error_message)
            out = data.rechunk(chunks=(1, -1, -1)).map_blocks(
                func=self._fw_locked,
                dtype=data.dtype,
                chunks=(1, self._n, np.prod(self._N)),
                name="_fw",
                meta=data._meta,
                lock=lock,
            )
            out = out.reshape(-1, sh[-1])[:N].reshape(sh)
        else:
            blks = [self._fw(blk) for blk in data]
            out = self._postprocess(blks, N, sh)
        return out

    def _nudft_apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if isinstance(arr, da.Array):
            out = da.tensordot(arr, self.complex_matrix(pycu.get_array_module(arr)), axes=(-1, -1))
        else:
            out = _nudft1(
                w=arr,
                x=self._x,
                mesh=self.mesh().reshape(-1, self._D),
                isign=self._isign,
                dtype=pycrt.getPrecision().complex.value,
            )
        return out

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: pyct.NDArray (constructor-dependant)
            (...,  N.prod()) input weights :math:`\mathbf{u} \in
            \mathbb{R}^{\mathcal{I}_{N_1,\ldots, N_d}}`
            (..., 2N.prod()) input weights :math:`\mathbf{u} \in
            \mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` viewed as a real array.
            (see :py:func:`~pycsou.util.complex.view_as_real`.)

        Returns
        -------
        out: pyct.NDArray (constructor-dependant)
            (...,  M) output weights :math:`\mathbf{w} \in \mathbb{R}^{M}`.
            (..., 2M) output weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.
            (see :py:func:`~pycsou.util.complex.view_as_real`.)
        """
        if self._real_output:
            r_width = pycrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pycu.view_as_complex(arr)

        if self._eps > 0:
            out = self._nufft_adjoint(arr)
        else:
            out = self._nudft_adjoint(arr)

        return out.real if self._real_input else pycu.view_as_real(out)

    def _nufft_adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        data, N, sh = self._preprocess(arr, self._n, self._M)

        if isinstance(data, da.Array):
            try:
                lock = dad.Lock()  # Use lock to avoid race condition because of the shared FFTW resources.
            except:
                error_message = r"""
                        The NUFFT operator requires Dask's distributed scheduler in multi-threading mode (other schedulers are not accepted).
                        Start a client and point it to the scheduler address:
                            from dask.distributed import Client
                            client = Client('ip-addr-of-scheduler:8786', processes=False)
                        """
                raise ValueError(error_message)
            out = data.rechunk(chunks=(1, -1, -1)).map_blocks(
                func=self._bw_locked,
                dtype=data.dtype,
                chunks=(1, self._n, self._M),
                name="_bw",
                meta=data._meta,
                lock=lock,
            )
            out = out.reshape(-1, sh[-1])[:N].reshape(sh)
        else:
            blks = [self._bw(blk) for blk in data]
            out = self._postprocess(blks, N, sh)
        return out

    def _nudft_adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        if isinstance(arr, da.Array):
            out = da.tensordot(arr, self.complex_matrix(pycu.get_array_module(arr)).conj().T, axes=(-1, -1))
        else:
            out = _nudft2(
                w=arr,
                x=self._x,
                mesh=self.mesh().reshape(-1, self._D),
                isign=-self._isign,
                dtype=pycrt.getPrecision().complex.value,
            )
        return out

    @pycrt.enforce_precision()
    def mesh(self, xp: pyct.ArrayModule = np) -> pyct.NDArray:
        mesh = xp.stack(xp.meshgrid(*[xp.arange(-(m // 2), (m - 1) // 2 + 1) for m in self._N], indexing="ij"), axis=-1)
        if self._modeord:
            mesh = xp.stack([xp.fft.ifftshift(mesh[..., i], axes=i) for i in range(mesh.shape[-1])], axis=-1)
        return mesh

    def complex_matrix(self, xp: pyct.ArrayModule = np) -> pyct.NDArray:
        A = self.mesh(xp)
        A = A.reshape((-1, self._D))
        A = xp.exp(1j * xp.sign(self._isign) * A @ self._x.T).astype(pycrt.getPrecision().complex.value)
        return A

    def asarray(
        self,
        xp: pyct.ArrayModule = np,
        dtype: typ.Optional[type] = None,
    ) -> pyct.NDArray:
        if dtype is None:
            dtype = pycrt.getPrecision().value

        try:
            width = pycrt.Width(np.dtype(dtype))
        except:
            raise ValueError(f"Unsupported dtype {dtype}.")

        cmat = self.complex_matrix(xp=xp).astype(width.complex.value)
        return pycu.view_as_real_mat(cmat, real_input=self._real_input, real_output=self._real_output).astype(dtype)


class _NUFFT3(NUFFT):
    def __init__(self, **kwargs):
        self._real = kwargs.pop("real")
        self._plan = dict(
            fw=self._plan_fw(**kwargs),
            bw=self._plan_bw(**kwargs),
        )
        self._M, self._D = kwargs["x"].shape  # Useful constants
        self._N, _ = kwargs["z"].shape
        self._n = self._plan["fw"].n_trans
        self._isign = kwargs["isign"]
        self._x = kwargs["x"]
        self._z = kwargs["z"]

        sh_op = [2 * self._N, 2 * self._M]
        if self._real:
            sh_op[1] //= 2
        super().__init__(shape=sh_op)
        self._lipschitz = np.sqrt(self._N * self._M)

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        kwargs = kwargs.copy()
        for k in ("nufft_type", "n_modes_or_dim", "dtype"):
            kwargs.pop(k, None)
        x = kwargs["x"] = cls._as_canonical_coordinate(kwargs["x"])
        z = kwargs["z"] = cls._as_canonical_coordinate(kwargs["z"])
        assert x.shape[-1] == z.shape[-1], "x vs. z: dimensionality mis-match."
        assert pycu.get_array_module(x) == pycu.get_array_module(z)
        kwargs["isign"] = int(np.sign(kwargs["isign"]))
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
                    tuple(_.astype(pycrt.getPrecision().value) for _ in pycu.compute(*x.T[:N_dim], *z.T[:N_dim])),
                    # astype() is needed here because dask.distributed passes a dtype that FINUFFT does not recognize as the builtin np.dtype('float64') (== passes but not "is" which FINUFFT uses)
                )
            ),
        )
        return plan

    def _fw(self, arr: np.ndarray) -> np.ndarray:
        out = self._plan["fw"].execute(arr.squeeze())  # ([n_trans], M) -> ([n_trans], N)
        return out.reshape((self._n, self._N))

    def _fw_locked(self, arr: np.ndarray, lock: dad.Lock = None) -> np.ndarray:
        arr = arr.astype(
            pycrt.getPrecision().complex.value
        )  # astype() is needed here because dask.distributed passes a dtype that FINUFFT does not recognize as the builtin np.dtype('complex128') (== passes but not "is" which FINUFFT uses)
        with lock:
            out = self._plan["fw"].execute(arr.squeeze())  # ([n_trans], M) -> ([n_trans], N)
        return out.reshape((self._n, self._N))[
            None, ...
        ]  # with map_blocks we can skip _postprocess by simply recreating the stacking axis after processing each chunk

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
                    tuple(_.astype(pycrt.getPrecision().value) for _ in pycu.compute(*z.T[:N_dim], *x.T[:N_dim])),
                    # astype() is needed here because dask.distributed passes a dtype that FINUFFT does not recognize as the builtin np.dtype('float64') (== passes but not "is" which FINUFFT uses)
                )
            ),
        )
        return plan

    def _bw(self, arr: np.ndarray) -> np.ndarray:
        out = self._plan["bw"].execute(arr.squeeze())  # ([n_trans,] N) -> ([n_trans,] M)
        return out.reshape((self._n, self._M))

    def _bw_locked(self, arr: np.ndarray, lock: dad.Lock = None) -> np.ndarray:
        arr = arr.astype(
            pycrt.getPrecision().complex.value
        )  # astype() is needed here because dask.distributed passes a dtype that FINUFFT does not recognize as the builtin np.dtype('complex128') (== passes but not "is" which FINUFFT uses)
        with lock:
            out = self._plan["bw"].execute(arr.squeeze())  # ([n_trans], N) -> ([n_trans], M)
        return out.reshape((self._n, self._M))[
            None, ...
        ]  # with map_blocks we can skip _postprocess by simply recreating the stacking axis after processing each chunk

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: pyct.NDArray (constructor-dependant)
            (...,  M) input weights :math:`\mathbf{w} \in \mathbb{R}^{M}`.
            (..., 2M) input weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.
            (see :py:func:`~pycsou.util.complex.view_as_real`.)

        Returns
        -------
        out: pyct.NDArray
            (..., 2N) output weights :math:`\mathbf{v} \in \mathbb{C}^{N}` viewed as a real array.
            (see :py:func:`~pycsou.util.complex.view_as_real`.)
        """
        if self._real:
            r_width = pycrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pycu.view_as_complex(arr)
        data, N, sh = self._preprocess(arr, self._n, self._N)

        if isinstance(data, da.Array):
            try:
                lock = dad.Lock()  # Use lock to avoid race condition because of the shared FFTW resources.
            except:
                error_message = r"""
                The NUFFT operator requires Dask's distributed scheduler in multi-threading mode (other schedulers are not accepted).
                Start a client and point it to the scheduler address:
                    from dask.distributed import Client
                    client = Client('ip-addr-of-scheduler:8786', processes=False)
                """
                raise ValueError(error_message)
            out = data.rechunk(chunks=(1, -1, -1)).map_blocks(
                func=self._fw_locked,
                dtype=data.dtype,
                chunks=(1, self._n, self._N),
                name="_fw",
                meta=data._meta,
                lock=lock,
            )
            out = out.reshape(-1, sh[-1])[:N].reshape(sh)
        else:
            blks = [self._fw(blk) for blk in data]
            out = self._postprocess(blks, N, sh)
        return pycu.view_as_real(out)

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: pyct.NDArray
            (..., 2N) input weights :math:`\mathbf{v} \in \mathbb{C}^{N}` viewed as a real array.
            (see :py:func:`~pycsou.util.complex.view_as_real`.)

        Returns
        -------
        out: pyct.NDArray (constructor-dependant)
            (...,  M) output weights :math:`\mathbf{w} \in \mathbb{R}^{M}`.
            (..., 2M) output weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.
            (see :py:func:`~pycsou.util.complex.view_as_real`.)
        """
        arr = pycu.view_as_complex(arr)
        data, N, sh = self._preprocess(arr, self._n, self._M)
        if isinstance(data, da.Array):
            try:
                lock = dad.Lock("bw_lock")  # Use lock to avoid race condition because of the shared FFTW resources.
            except:
                error_message = r"""
                The NUFFT operator requires Dask's distributed scheduler in multi-threading mode (other schedulers are not accepted).
                Start a client and point it to the scheduler address:
                    from dask.distributed import Client
                    client = Client('ip-addr-of-scheduler:8786', processes=False)
                """
                raise ValueError(error_message)
            out = data.rechunk(chunks=(1, -1, -1)).map_blocks(
                func=self._bw_locked,
                dtype=data.dtype,
                chunks=(1, self._n, self._M),
                name="_bw",
                meta=data._meta,
                lock=lock,
            )
            out = out.reshape(-1, sh[-1])[:N].reshape(sh)
        else:
            blks = [self._bw(blk) for blk in data]
            out = self._postprocess(blks, N, sh)
        return out.real if self._real else pycu.view_as_real(out)

    def complex_matrix(self, xp: pyct.ArrayModule = np) -> pyct.NDArray:
        return xp.exp(1j * xp.sign(self._isign) * self._z @ self._x.T).astype(pycrt.getPrecision().complex.value)

    def asarray(
        self,
        xp: pyct.ArrayModule = np,
        dtype: typ.Optional[type] = None,
    ) -> pyct.NDArray:
        if dtype is None:
            dtype = pycrt.getPrecision().value

        try:
            width = pycrt.Width(np.dtype(dtype))
        except:
            raise ValueError(f"Unsupported dtype {dtype}.")

        cmat = self.complex_matrix(xp=xp).astype(width.complex.value)
        return pycu.view_as_real_mat(cmat, real_input=self._real).astype(dtype)


@nb.njit(parallel=True, fastmath=True, nogil=True)
def _nudft1_cpu(
    w: pyct.NDArray, x: pyct.NDArray, mesh: pyct.NDArray, *, isign: typ.Literal[1, -1], dtype: np.dtype
) -> pyct.NDArray:
    out = np.zeros(w.shape[:-1] + (mesh.shape[0],), dtype=dtype)
    for n in nb.prange(mesh.shape[0]):
        for j in nb.prange(x.shape[0]):
            out[..., n] += w[..., j] * np.exp(isign * 1j * np.dot(x[j, :], mesh[n, :]))
    return out


def _nudft1_gpu():
    pass


@pycu.redirect("w", NUMPY=_nudft1_cpu, CUPY=_nudft1_gpu)
def _nudft1(
    w: pyct.NDArray, x: pyct.NDArray, mesh: pyct.NDArray, *, isign: typ.Literal[1, -1], dtype: np.dtype
) -> pyct.NDArray:
    raise NotImplementedError


@nb.njit(parallel=True, fastmath=True, nogil=True)
def _nudft2_cpu(
    w: pyct.NDArray, x: pyct.NDArray, mesh: pyct.NDArray, isign: typ.Literal[1, -1], dtype: np.dtype
) -> pyct.NDArray:
    out = np.zeros(w.shape[:-1] + (x.shape[0],), dtype=dtype)
    for j in nb.prange(x.shape[0]):
        for n in nb.prange(mesh.shape[0]):
            out[..., j] += w[..., n] * np.exp(isign * 1j * np.dot(x[j, :], mesh[n, :]))
    return out


def _nudft2_gpu():
    pass


@pycu.redirect("w", NUMPY=_nudft2_cpu, CUPY=_nudft2_gpu)
def _nudft2(
    w: pyct.NDArray, x: pyct.NDArray, mesh: pyct.NDArray, isign: typ.Literal[1, -1], dtype: np.dtype
) -> pyct.NDArray:
    pass


def _nudft3():
    pass
