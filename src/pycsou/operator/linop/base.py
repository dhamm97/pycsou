import types
import typing as typ
import warnings

import numpy as np
import scipy.sparse as scisp
import sparse as sp

import pycsou.abc.operator as pyco
import pycsou.math.stencil as pycstencil
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

if pycd.CUPY_ENABLED:
    import cupy as cp
    import cupyx.scipy.sparse as cusp

import math
import numbers
from typing import Callable


class ExplicitLinFunc(pyco.LinFunc):
    r"""
    Build a linear functional from its vectorial representation.

    Given a vector :math:`\mathbf{z}\in\mathbb{R}^N`, the *explicit linear functional* associated to  :math:`\mathbf{z}` is defined as

    .. math::

        f_\mathbf{z}(\mathbf{x})=\mathbf{z}^T\mathbf{x}, \qquad \forall \mathbf{x}\in\mathbb{R}^N,

    with adjoint given by:

    .. math::

        f^\ast_\mathbf{z}(\alpha)=\alpha\mathbf{z}, \qquad \forall \alpha\in\mathbb{R}.

    The vector :math:`\mathbf{z}` is called the *vectorial representation* of the linear functional :math:`f_\mathbf{z}`.
    The lipschitz constant of explicit linear functionals is trivially given by :math:`\|\mathbf{z}\|_2`.

    Examples
    --------
    >>> from pycsou.operator.linop.base import ExplicitLinFunc
    >>> import numpy as np
    >>> vec = np.ones(10)
    >>> sum_func = ExplicitLinFunc(vec)
    >>> sum_func.shape
    (1, 10)
    >>> np.allclose(sum_func(np.arange(10)), np.sum(np.arange(10)))
    True
    >>> np.allclose(sum_func.adjoint(3), 3 * vec)
    True

    Notes
    -----
    :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc` instances are **not array module agnostic**: they will only work with input arrays
    belonging to the same array module than the one of the array ``vec`` used to initialize the :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc` object.
    Moreover, while the input/output precisions of the callable methods of :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc` objects are
    guaranteed to match the user-requested precision, the inner computations may force a recast of the input arrays when
    the precision of ``vec`` does not match the user-requested precision. If such a situation occurs, a warning is raised.

    See Also
    --------
    :py:meth:`~pycsou.abc.operator.LinOp.asarray`
        Convert a matrix-free :py:class:`~pycsou.abc.operator.LinFunc` into an :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc`.
    """

    @pycrt.enforce_precision(i="vec")
    def __init__(self, vec: pyct.NDArray, enable_warnings: bool = True):
        r"""

        Parameters
        ----------
        vec: NDArray
            (N,) input. N-D input arrays are flattened. This is the vectorial representation of the linear functional.
        enable_warnings: bool
            If ``True``, the user will be warned in case of mismatching precision issues.

        Notes
        -----
        The input ``vec`` is automatically casted by the decorator :py:func:`~pycsou.runtime.enforce_precision` to the user-requested precision at initialization time.
        Explicit control over the precision of ``vec`` is hence only possible via the context manager :py:class:`~pycsou.runtime.Precision`:

        >>> from pycsou.operator.linop.base import ExplicitLinFunc
        >>> import pycsou.runtime as pycrt
        >>> import numpy as np
        >>> vec = np.ones(10) # This array will be recasted to requested precision.
        >>> with pycrt.Precision(pycrt.Width.HALF):
        ...     sum_func = ExplicitLinFunc(vec) # The init function of ExplicitLinFunc stores ``vec`` at the requested precision.
        ...     # Further calculations with sum_func. Within this context mismatching precisions are avoided.

        """
        super(ExplicitLinFunc, self).__init__(shape=(1, vec.size))
        xp = pycu.get_array_module(vec)
        self.vec = vec.copy().reshape(-1)
        self._lipschitz = xp.linalg.norm(vec)
        self._enable_warnings = bool(enable_warnings)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self.vec.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return (self.vec * arr).sum(axis=-1)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self.vec.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return arr * self.vec

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        if self.vec.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return xp.broadcast_to(self.vec, arr.shape)

    @pycrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        if self.vec.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        return arr - tau * self.vec


class IdentityOp(pyco.PosDefOp, pyco.UnitOp):
    r"""
    Identity operator :math:`\mathrm{Id}`.
    """

    def __init__(self, shape: pyct.SquareShape):
        pyco.PosDefOp.__init__(self, shape)
        pyco.UnitOp.__init__(self, shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr


class HomothetyOp(pyco.SelfAdjointOp):
    r"""
    Scaling operator.

    This operators performs a scaling by real factor ``cst``. Its Lipschitz constant is given by ``abs(cst)``.
    """

    def __init__(self, cst: pyct.Real, dim: int):
        r"""

        Parameters
        ----------
        cst: Real
            Scaling factor.
        dim: int
            Dimension of the domain.

        Raises
        ------
        ValueError
            If ``cst`` is not real.
        """
        if not isinstance(cst, pyct.Real):
            raise ValueError(f"cst: expected real number, got {cst}.")
        super(HomothetyOp, self).__init__(shape=(dim, dim))
        self._cst = cst
        self._lipschitz = abs(cst)
        self._diff_lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        out = arr.copy()
        out *= self._cst
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    def __mul__(self, other):
        op = pyco.Property.__mul__(self, other)
        if isinstance(other, pyco.ProxFunc):
            op.specialize(cast_to=pyco.ProxFunc)
            post_composition_prox = lambda obj, arr, tau: other.prox(arr, self._cst * tau)
            op.prox = types.MethodType(post_composition_prox, op)
        return op


class NullOp(pyco.LinOp):
    r"""
    Null operator.

    This operator maps any input vector on the null vector. Its Lipschitz constant is zero.
    """

    def __init__(self, shape: typ.Tuple[int, int]):
        r"""

        Parameters
        ----------
        shape: tuple(int, int)
            Shape of the null operator.
        """
        super(NullOp, self).__init__(shape)
        self._lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[:-1], self.codim),
        )

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (arr.shape[:-1], self.dim),
        )


class NullFunc(NullOp, pyco.LinFunc):
    r"""
    Null functional.

    This functional maps any input vector on the null scalar. Its Lipschitz constant is zero.
    """

    def __init__(self, dim: typ.Optional[int] = None):
        r"""

        Parameters
        ----------
        dim: Optional[int]
            Dimension of the domain. Set ``dim=None`` for making the functional domain-agnostic.
        """
        pyco.LinFunc.__init__(self, shape=(1, dim))
        NullOp.__init__(self, shape=self.shape)

    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    @pycrt.enforce_precision(i="arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return arr


class ExplicitLinOp(pyco.LinOp):
    r"""
    Build a linear operator from its matrix representation.

    Given a matrix :math:`\mathbf{A}\in\mathbb{R}^{M\times N}`, the *explicit linear operator* associated to  :math:`\mathbf{A}` is defined as

    .. math::

        f_\mathbf{A}(\mathbf{x})=\mathbf{A}\mathbf{x}, \qquad \forall \mathbf{x}\in\mathbb{R}^N,

    with adjoint given by:

    .. math::

        f^\ast_\mathbf{A}(\mathbf{z})=\mathbf{A}^T\mathbf{z}, \qquad \forall \mathbf{z}\in\mathbb{R}^M.

    Examples
    --------
    >>> from pycsou.operator.linop.base import ExplicitLinOp
    >>> import numpy as np
    >>> mat = np.arange(10).reshape(2,5)
    >>> f = ExplicitLinOp(mat)
    >>> f.shape
    (2, 5)
    >>> np.allclose(f(np.arange(5)), mat @ np.arange(5))
    True
    >>> np.allclose(f.adjoint(np.arange(2)), mat.T @ np.arange(2))
    True

    Notes
    -----
    :py:class:`~pycsou.operator.linop.base.ExplicitLinOp` instances are **not array module agnostic**: they will only work with input arrays
    belonging to the same array module than the one of the array ``mat`` used to initialize the :py:class:`~pycsou.operator.linop.base.ExplicitLinOp` object.
    Moreover, while the input/output precisions of the callable methods of :py:class:`~pycsou.operator.linop.base.ExplicitLinOp` objects are
    guaranteed to match the user-requested precision, the inner computations may force a recast of the input arrays when
    the precision of ``mat`` does not match the user-requested precision. If such a situation occurs, a warning is raised.

    See Also
    --------
    :py:class:`~pycsou.operator.linop.base.ExplicitLinFunc`
    :py:meth:`~pycsou.abc.operator.LinOp.asarray`
        Convert a matrix-free :py:class:`~pycsou.abc.operator.LinOp` into an :py:class:`~pycsou.operator.linop.base.ExplicitLinOp`.
    """

    @pycrt.enforce_precision(i="mat")
    def __init__(self, mat: typ.Union[pyct.NDArray, pyct.SparseArray], enable_warnings: bool = True):
        r"""

        Parameters
        ----------
        mat: NDArray | SparseArray
            (M,N) input array. This is the matrix representation of the linear operator. The input array can be *dense* or *sparse*.
            In the latter case, it must be an instance of one of the following sparse array classes: :py:class:`sparse.SparseArray`,
            :py:class:`scipy.sparse.spmatrix`, :py:class:`cupyx.scipy.sparse.spmatrix`. Note that
        enable_warnings: bool
            If ``True``, the user will be warned in case of mismatching precision issues.

        Notes
        -----
        The input ``mat`` is automatically casted by the decorator :py:func:`~pycsou.runtime.enforce_precision` to the user-requested precision at initialization time.
        Explicit control over the precision of ``mat`` is hence only possible via the context manager :py:class:`~pycsou.runtime.Precision`:

        >>> from pycsou.operator.linop.base import ExplicitLinOp
        >>> import pycsou.runtime as pycrt
        >>> import numpy as np
        >>> mat = np.arange(10).reshape(2,5) # This array will be recasted to requested precision.
        >>> with pycrt.Precision(pycrt.Width.HALF):
        ...     f = ExplicitLinOp(mat) # The init function of ExplicitLinOp stores ``mat`` at the requested precision.
        ...     # Further calculations with f. Within this context mismatching precisions are avoided.

        Note moreover that sparse inputs with type :py:class:`scipy.sparse.spmatrix` are automatically casted as :py:class:`sparse.SparseArray` which should be
        the preferred class for representing sparse arrays. Finally, the default sparse storage format is ``'csr'`` (for fast matrix-vector multiplications).
        """
        super(ExplicitLinOp, self).__init__(shape=mat.shape)
        self.mat = self._coerce_input(mat)
        self._enable_warnings = bool(enable_warnings)
        self._module_name = self._get_module_name(mat)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self.mat.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        if self._module_name == "cupyx":
            stack_shape = arr.shape[:-1]
            return cp.asarray(self.mat.dot(arr.reshape(-1, self.dim).transpose()).transpose()).reshape(
                *stack_shape, self.codim
            )
        else:
            return self.mat.__matmul__(arr[..., None]).squeeze(axis=-1)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        if self.mat.dtype != pycrt.getPrecision().value and self._enable_warnings:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)
        if self._module_name == "cupyx":
            stack_shape = arr.shape[:-1]
            return cp.asarray(self.mat.T.dot(arr.reshape(-1, self.dim).transpose()).transpose()).reshape(
                *stack_shape, self.codim
            )
        else:
            return self.mat.transpose().__matmul__(arr[..., None]).squeeze(axis=-1)

    def lipschitz(self, recompute: bool = False, algo: str = "svds", **kwargs) -> float:
        r"""
        Same functionality as :py:meth:`~pycsou.abc.operator.LinOp.lipschitz` but the case ``algo='fro'`` is handled
        differently: the Frobenius norm of the operator is directly computed from its matrix representation rather than with the Hutch++ algorithm.
        """
        kwargs.pop("gpu", None)
        gpu = True if self._module_name in ["cupy", "cupyx"] else False
        if recompute or (self._lipschitz == np.inf):
            if algo == "fro":
                if self._module_name in ["sparse", "cupyx"]:
                    data = self.mat.asformat("coo").data.squeeze()
                    xp = pycu.get_array_module(data)
                    self._lipschitz = xp.linalg.norm(data, ord=algo)
                else:
                    xp = pycu.get_array_module(self.mat)
                    self._lipschitz = xp.linalg.norm(self.mat, ord=algo)
            else:
                self._lipschitz = pyco.LinOp.lipschitz(self, recompute=recompute, algo=algo, gpu=gpu, **kwargs)
        return self._lipschitz

    def svdvals(self, k: int, which="LM", **kwargs) -> pyct.NDArray:
        kwargs.pop("gpu", None)
        gpu = True if self._module_name in ["cupy", "cupyx"] else False
        return pyco.LinOp.svdvals(self, k=k, which=which, gpu=gpu, **kwargs)

    def _coerce_input(
        self, mat: typ.Union[pyct.NDArray, pyct.SparseArray]
    ) -> typ.Union[pyct.NDArray, pyct.SparseArray]:
        assert type(mat) in pycd.supported_array_types() + pycd.supported_sparse_types()
        if pycd.CUPY_ENABLED and isinstance(mat, cusp.spmatrix):
            out = mat.tocsr(copy=True)
        elif isinstance(mat, scisp.spmatrix):
            out = sp.GCXS.from_scipy_sparse(mat)
        elif isinstance(mat, sp.SparseArray):
            assert mat.ndim == 2
            out = mat.asformat("gcxs")
        else:
            assert mat.ndim == 2
            out = mat.copy()
        return out

    def _get_module_name(self, arr: typ.Union[pyct.NDArray, pyct.SparseArray]) -> str:
        if pycd.CUPY_ENABLED and isinstance(arr, cusp.spmatrix):
            return "cupyx"
        else:
            array_module = pycu.get_array_module(arr, fallback=sp)
            return array_module.__name__

    def trace(self, **kwargs) -> float:
        return self.mat.trace().item()


class Stencil(pyco.SquareOp):
    r"""
    Base class for NDArray computing functions that operate only on a local region of the NDArray through a
    multi-dimensional kernel, namely through correlation and convolution.

    This class leverages the :py:func:`numba.stencil` decorator, which allowing to JIT (Just-In-Time) compile these
    functions to run more quickly.

    Examples
    --------
    The following example creates a Stencil operator based on a 2-dimensional kernel. It shows how to perform correlation
    and convolution in CPU, GPU (Cupy) and distributed across different cores (Dask).

    .. code-block:: python3
        from pycsou.operator.linop.base import Stencil
        import numpy as np
        import cupy as cp
        import dask.array as da

        nsamples = 2
        data_shape = (5, 10)

        # Numpy
        data = np.ones((nsamples, *data_shape)).reshape(nsamples, -1)
        # Cupy
        data_cu = cp.ones((nsamples, *data_shape)).reshape(nsamples, -1)
        # Dask
        data_da = da.ones((nsamples, *data_shape)).reshape(nsamples, -1)

        kernel = np.array([[0.5, 0, 0.5], [0, 0, 0], [0.5, 0, 0.5]])
        center = np.array([1, 0])

        stencil = Stencil(stencil_coefs=kernel, center=center, arg_shape=data_shape, mode=0.)
        stencil_cu = Stencil(stencil_coefs=cp.asarray(kernel), center=center, arg_shape=data_shape, mode=0.)

        # Correlate images with kernels
        out = stencil(data).reshape(nsamples, *data_shape)
        out_da = stencil(data_da).reshape(nsamples, *data_shape).compute()
        out_cu = stencil_cu(data_cu).reshape(nsamples, *data_shape).get()

        # Convolve images with kernels
        out_adj = stencil.adjoint(data).reshape(nsamples, *data_shape)
        out_da_adj = stencil.adjoint(data_da).reshape(nsamples, *data_shape).compute()
        out_cu_adj = stencil_cu.adjoint(data_cu).reshape(nsamples, *data_shape).get()

    Remark 1
    --------
    The :py:class:`~pycsou.operator.linop.base.Stencil` allows to perform both correlation and convolution. By default,
    the ``apply`` method will perform **correlation** of the input array with the given kernel / stencil, whereas the
    ``adjoint`` method will perform **convolution**.

    Remark 2
    --------
    There are five padding mode supported: ‘reflect’, ‘periodic’, ‘nearest’, ‘none’ (zero padding), or 'cval'
    (constant value). See `Dask's padding options <https://docs.dask.org/en/stable/array-overlap.html#boundaries>`_ to deal with
    overlapping computations for a more detailed explanation.

    Remark 3
    --------
    When applying performing a stencil computations with :py:func:`~Dask.array.map_overlap`, if the kernel is asymmetric,
    then the only padding option allowed is 'none'. If a different padding option is desired, and Dask is
    used, then a new symmetric kernel (with padded with zeros) will be created.

    Remark 4
    --------
    By default, for GPU computing, the ``threadsperblock`` argument is set according to:

    .. math::

        \prod_{i=1}^{D} t_{i} \leq c

    where :math:`D` is the number of dimensions of the input, and  :math:`c=1024` is the
    `limit number of threads per block in current GPUs <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>`_.
    """

    def __init__(
        self,
        stencil_coefs: pyct.NDArray,
        center: pyct.NDArray,
        arg_shape: pyct.Shape,
        is_normal: bool = False,
        **kwargs: typ.Optional[dict],
    ):
        r"""

        Parameters
        ----------
        stencil_coefs: NDArray
            Stencil coefficients. Must have the same number of dimension as the input array's arg_shape (i.e., without the
            stacking dimension).
        center: NDArray
            Index of the kernel's center. Must be a 1-dimensional array with one element per dimension in ``stencil_coefs``.
        arg_shape: tuple
            Shape of the input array
        is_normal: bool
            Whether the resulting linear operator corresponds to a :py:class:`~pycsou.abc.operator.NormalOp`.
        kwargs
            Extra kwargs for `padding control <https://docs.dask.org/en/stable/array-overlap.html#boundaries>`_,
            for `Numba's just-in-time compilation
            <https://numba.readthedocs.io/en/stable/reference/jit-compilation.html?highlight=nogil>`_ (supported
            arguments are ``parallel``, ``fastmath`` and ``nogil``, which are all used by default), and finally, the GPU
            option ``threadsperblock``
        """
        size = np.prod(arg_shape).item()

        if is_normal:
            # TODO: CHANGE CLASS TO NormalOp (with __new__?)
            pass

        super(Stencil, self).__init__((size))
        stencil_kwargs = dict(
            [(key, kwargs.pop(key)) for key in list(kwargs.keys()) if key in ["parallel", "fastmath", "nogil"]]
        )
        self.arg_shape = arg_shape
        self.ndim = len(arg_shape)
        self._check_stencil_inputs(stencil_coefs, center, **kwargs)
        self._make_stencils(self.stencil_coefs, **stencil_kwargs)

    def _apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._postprocess(self.stencil(self._preprocess(arr)), out_shape=arr.shape)

    def _apply_dask(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr.reshape(-1, *self.arg_shape).map_overlap(
            self.stencil_dask, depth=self._depth, boundary=self._boundaries, dtype=pycrt.getPrecision().value
        )

    def _apply_cupy(self, arr: pyct.NDArray) -> pyct.NDArray:
        out_shape = arr.shape
        arr, out = self._allocate_output_preproc(arr)
        blockspergrid = [math.ceil(out.shape[i] / tpb) for i, tpb in enumerate(self.threadsperblock)]
        self.stencil[blockspergrid, self.threadsperblock](arr, out)
        return self._postprocess(out, out_shape)

    def _adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._postprocess(
            self.stencil_adjoint(self._preprocess(arr, direction="adjoint")), out_shape=arr.shape, direction="adjoint"
        )

    def _adjoint_dask(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr.reshape(-1, *self.arg_shape).map_overlap(
            self.stencil_adjoint_dask,
            depth=self._depth_adjoint,
            boundary=self._boundaries,
            dtype=pycrt.getPrecision().value,
        )

    def _adjoint_cupy(self, arr: pyct.NDArray) -> pyct.NDArray:
        out_shape = arr.shape
        arr, out = self._allocate_output_preproc(arr, direction="adjoint")
        blockspergrid = tuple([math.ceil(arr.shape[i] / tpb) for i, tpb in enumerate(self.threadsperblock)])
        self.stencil_adjoint[blockspergrid, self.threadsperblock](arr, out)
        return self._postprocess(out, out_shape, direction="adjoint")

    @pycrt.enforce_precision(i="arr")
    @pycu.redirect("arr", DASK=_apply_dask, CUPY=_apply_cupy)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Array to be correlated with the kernel.

        Returns
        -------
        out: NDArray
            NDArray with same shape as the input NDArray, correlated with kernel.
        """
        return self._apply(arr)

    @pycrt.enforce_precision(i="arr")
    @pycu.redirect("arr", DASK=_adjoint_dask, CUPY=_adjoint_cupy)
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Array to be convolved with the kernel.

        Returns
        -------
        out: NDArray
            NDArray with same shape as the input NDArray, convolved with kernel.
        """
        return self._adjoint(arr)

    def _make_stencils_cpu(self, stencil_coefs: pyct.NDArray, **kwargs) -> None:
        self.stencil = pycstencil.make_nd_stencil(self.stencil_coefs, self.center, **kwargs)
        self.stencil_dask = pycstencil.make_nd_stencil(self.stencil_coefs_dask, self.center_dask, **kwargs)
        self.stencil_adjoint = pycstencil.make_nd_stencil(self.stencil_coefs_adjoint, self.center_adjoint, **kwargs)
        self.stencil_adjoint_dask = pycstencil.make_nd_stencil(
            self.stencil_coefs_adjoint_dask, self.center_adjoint_dask, **kwargs
        )

    def _make_stencils_gpu(self, stencil_coefs: pyct.NDArray, **kwargs) -> None:
        self.stencil = pycstencil.make_nd_stencil_gpu(self.stencil_coefs, self.center)
        self.stencil_dask = pycstencil.make_nd_stencil(self.stencil_coefs_dask, self.center_dask)
        self.stencil_adjoint = pycstencil.make_nd_stencil_gpu(self.stencil_coefs_adjoint, self.center_adjoint)
        self.stencil_adjoint_dask = pycstencil.make_nd_stencil(
            self.stencil_coefs_adjoint_dask, self.center_adjoint_dask
        )

    @pycu.redirect("stencil_coefs", CUPY=_make_stencils_gpu)
    def _make_stencils(self, stencil_coefs: pyct.NDArray, **kwargs) -> None:
        self._make_stencils_cpu(stencil_coefs, **kwargs)

    def _preprocess(self, arr: pyct.NDArray, direction: str = "apply") -> pyct.NDArray:
        r"""
        Pad input according to the kernel's shape and center.
        """
        xp = pycu.get_array_module(arr)
        arr = arr.reshape(-1, *self.arg_shape)
        for i in range(1, len(self.pad_withs[direction])):
            padding_kwargs = {key: value[i] for key, value in self._padding_kwargs.items()}
            _pad_width = tuple(
                [(0, 0) if i != j else self.pad_withs[direction][i] for j in range(len(self.pad_withs[direction]))]
            )
            arr = xp.pad(array=arr, pad_width=_pad_width, **padding_kwargs)
        return arr

    def _postprocess(self, arr: pyct.NDArray, out_shape: pyct.Shape, direction: str = "apply") -> pyct.NDArray:
        r"""
        Unpad the output of the correlation/convolution to have shape (stacking_dims, *arg_shape).
        """
        return self._unpad(arr, direction=direction).reshape(out_shape)

    def _check_stencil_inputs(self, stencil_coefs: pyct.NDArray, center: pyct.NDArray, **kwargs):
        r"""
        Check that inputs have the correct shape and correctly handle the boundary conditions.
        """
        assert len(center) == stencil_coefs.ndim == self.ndim, (
            "The stencil coefficients should have the same"
            " number of dimensions as `arg_shape` and the "
            "same length as `center`."
        )
        xp = pycu.get_array_module(stencil_coefs)
        self.stencil_coefs = self.stencil_coefs_dask = stencil_coefs
        self.center = self.center_dask = xp.asarray(center)
        self.stencil_coefs_adjoint = self.stencil_coefs_adjoint_dask = xp.flip(stencil_coefs)
        self.center_adjoint = self.center_adjoint_dask = xp.array(stencil_coefs.shape) - 1 - xp.asarray(center)

        ndim = stencil_coefs.ndim
        mode = kwargs.get("mode", "none")
        mode2 = dict()
        cval = dict()
        self.threadsperblock = kwargs.get("threadsperblock", [1] + [int(np.power(1024, 1 / (ndim)))] * (ndim))

        assert len(self.threadsperblock) == ndim + 1, (
            "`threadsperblock` must be a list with as many elements as "
            "kernel dimensions plus one (for stacking dimension)"
        )

        if not isinstance(mode, dict):
            mode = {i: mode for i in range(ndim + 1)}

        for i in range(ndim + 1):

            this_mode = mode.get(i, "none")
            if this_mode == "none":
                mode2.update(dict([(i, "constant")]))
                cval.update(dict([(i, 0.0)]))

            elif this_mode == "periodic":
                mode2.update(dict([(i, "wrap")]))

            elif this_mode == "reflect":
                mode2.update(dict([(i, "reflect")]))

            elif this_mode == "nearest":
                mode2.update(dict([(i, "edge")]))

            elif isinstance(this_mode, numbers.Number):
                mode2.update(dict([(i, "constant")]))
                cval.update(dict([(i, this_mode)]))
            else:
                raise ValueError(
                    f"`mode` should be `reflect`, `periodic`, `nearest`, `none` or a constant value,"
                    f" but got {this_mode} instead."
                )

        self._boundaries = mode
        self._padding_kwargs = dict(mode=mode2, constant_values=cval)
        depth_right = xp.array(self.stencil_coefs.shape) - self.center - 1
        _pad_width = tuple([(0, 0)] + [(self.center[i].item(), depth_right[i].item()) for i in range(ndim)])
        depth_right = xp.array(self.stencil_coefs_adjoint.shape) - self.center_adjoint - 1
        _pad_width_adjoint = tuple(
            [(0, 0)] + [(self.center_adjoint[i].item(), depth_right[i].item()) for i in range(ndim)]
        )
        self.pad_withs = dict(apply=_pad_width, adjoint=_pad_width_adjoint)
        # If boundary conditions are not 'none' for some dimension, then Dask's map_overlap needs a symmetric kernel.
        if any(map("none".__ne__, self._boundaries.values())):  # some key is not 'none' --> center dask kernel
            self._depth, self.stencil_coefs_dask, self.center_dask = self._convert_sym_ker(
                self.stencil_coefs_dask, self.center_dask
            )
            self._depth_adjoint, self.stencil_coefs_adjoint_dask, self.center_adjoint_dask = self._convert_sym_ker(
                self.stencil_coefs_adjoint_dask, self.center_adjoint_dask
            )
        else:
            depth_right = xp.array(self.stencil_coefs_dask.shape) - self.center_dask - 1
            self._depth = {0: 0}
            self._depth.update({i + 1: (self.center_dask[i], depth_right[i]) for i in range(self.ndim)})

            depth_right = xp.array(self.stencil_coefs_adjoint_dask.shape) - self.center_adjoint_dask - 1
            self._depth_adjoint = {0: 0}
            self._depth_adjoint.update({i + 1: (self.center_adjoint_dask[i], depth_right[i]) for i in range(self.ndim)})

    def _convert_sym_ker(
        self, stencil_coefs: pyct.NDArray, center: pyct.NDArray
    ) -> typ.Tuple[typ.Tuple, pyct.NDArray, pyct.NDArray]:
        r"""
        Creates a symmetric kernel stencil to use with Dask's map_overlap() in case of non-default ('none') boundary
        conditions.
        """
        xp = pycu.get_array_module(stencil_coefs)
        dist_center = (
            -(xp.array(stencil_coefs.shape) // 2 - xp.asarray(center)) * 2
            - xp.mod(xp.array(stencil_coefs.shape), 2)
            + 1
        )

        pad_left = abs(xp.clip(dist_center, a_min=-xp.infty, a_max=0)).astype(int)
        pad_right = xp.clip(dist_center, a_min=0, a_max=xp.infty).astype(int)
        pad_width = tuple([(pad_left[i].item(), pad_right[i].item()) for i in range(self.ndim)])
        stencil_coefs = xp.pad(stencil_coefs, pad_width=pad_width)
        center = xp.array(stencil_coefs.shape) // 2

        depth_sides = xp.array(stencil_coefs.shape) - center - 1
        _depth = tuple([0] + [depth_sides[i] for i in range(self.ndim)])
        return _depth, stencil_coefs, center

    def _unpad(self, arr: pyct.NDArray, direction: str = "apply") -> pyct.NDArray:
        slices = []
        for (start, end) in self.pad_withs[direction]:
            end = None if end == 0 else -end
            slices.append(slice(start, end))
        return arr[tuple(slices)]

    def _allocate_output_preproc(
        self, arr: pyct.NDArray, direction: str = "apply"
    ) -> typ.Tuple[pyct.NDArray, pyct.NDArray]:
        xp = pycu.get_array_module(arr)
        arr = self._preprocess(arr, direction=direction)
        out = xp.zeros_like(arr)
        return arr, out
