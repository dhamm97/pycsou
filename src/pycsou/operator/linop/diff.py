import collections.abc as cabc
import itertools
import math
import typing as typ

import numpy as np
import scipy.ndimage._filters as scif

import pycsou.abc.operator as pyco
import pycsou.compound as pycc
import pycsou.operator.linop.base
import pycsou.operator.linop.base as pycob
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = ["PartialDerivative", "Gradient", "Hessian", "Jacobian"]


class _Differential(pycob.Stencil):
    r"""
    Helper base class for differential operators based on Numba stencils (see
    https://numba.pydata.org/numba-doc/latest/user/stencil.html).

    See Also
    --------
    :py:class:`~pycsou.operator.linop.base.Stencil`, :py:func:`~pycsou.math.stencil.make_nd_stencil`,
    :py:class:`~pycsou.operator.linop.diff._FiniteDifferences`,
    :py:class:`~pycsou.operator.linop.diff._GaussianDerivative`,
    :py:class:`~pycsou.operator.linop.diff.PartialDerivative`, :py:class:`~pycsou.operator.linop.diff.Gradient`,
    :py:class:`~pycsou.operator.linop.diff.Hessian`.
    """

    def __init__(self, kernel, center, arg_shape, **kwargs):
        r"""
        Parameters
        ----------
        kernel: NDArray
        center: NDArray
        arg_shape: tuple
        **kwargs:
            Extra keyword parameters to :py:func:`math.stencil.make_nd_stencil`.
            Most useful are `mode` (padding) and `parallel` (stencil).
            In the case of computations on the GPU with CuPy arrays, `threadsperblock` is relevant.
        """

        super(_Differential, self).__init__(stencil_coefs=kernel, center=center, arg_shape=arg_shape, **kwargs)

    def _check_inputs(
        self,
        order: typ.Union[int, typ.Tuple[int, ...]],
        param1: typ.Union[str, float, typ.Tuple[str, ...], typ.Tuple[float, ...]],
        param1_name: str,
        param2: typ.Union[float, typ.Tuple[float, ...]],
        param2_name: str,
    ):
        r"""
        Checks that inputs have the appropriate shape and values.
        """
        self.order = self._ensure_tuple(order, param_name="order")
        self._param1 = self._ensure_tuple(param1, param_name="param1_name")
        self._param2 = self._ensure_tuple(param2, param_name="param2_name")

        assert all([o > 0 for o in self.order]), "Order must be strictly positive"

        if param1_name == "sigma":
            assert all([p >= 0 for p in self._param1]), "Sigma must be strictly positive"
        if param2_name == "accuracy":
            assert all([p >= 0 for p in self._param2]), "Accuracy must be positive"
        elif param2_name == "truncate":
            assert all([p > 0 for p in self._param2]), "Trimcate must be strictly positive"

        if len(self.order) != len(self.arg_shape):
            assert self.axis is not None, (
                "If `order` is not a tuple with size of arg_shape, then `axis` must be" " specified. Got `axis=None`"
            )
            self.axis = self._ensure_tuple(self.axis, param_name="axis")
            assert len(self.axis) == len(self.order), "`axis` must have the same number of elements as `order`"

        else:
            assert self.axis is None, (
                "If `order` is a tuple with size of arg_shape, then `axis` must not be" " specified (set `axis=None`)."
            )
            self.axis = tuple([i for i in range(len(self.arg_shape))])

        if not (len(self._param1) == len(self.order)):
            assert len(self._param1) == 1, (
                f"Parameter `{param1_name}` inconsistent with the number of elements in " "parameter `order`."
            )
            self._param1 = self._param1 * len(self.order)

        if not (len(self._param2) == len(self.order)):
            assert len(self._param2) == 1, (
                f"Parameter `{param2_name}` inconsistent with the number of elements in " "parameter `order`."
            )
            self._param2 = self._param2 * len(self.order)

    def _ensure_tuple(self, param, param_name: str) -> typ.Union[tuple[int, ...], tuple[str, ...]]:
        r"""
        Checks that inputs have the appropriate shape.
        """
        if not isinstance(param, cabc.Sequence) or isinstance(param, str):
            param = (param,)
        assert (len(param) == 1) | (len(param) == len(self.arg_shape)), (
            f"The length of {param_name} cannot be larger than the"
            f"number of dimensions ({len(self.arg_shape)}) defined by `arg_shape`"
        )
        return param

    def _create_kernel(self) -> typ.Tuple[pyct.NDArray, pyct.NDArray]:
        r"""
        Creates kernel for stencil.
        """
        stencil_ids = [None] * len(self.arg_shape)
        stencil_coefs = [None] * len(self.arg_shape)
        center = np.zeros(len(self.arg_shape), dtype=int)

        # Create finite difference coefficients for each dimension
        for i, ax in enumerate(self.axis):
            stencil_ids[ax], stencil_coefs[ax], center[ax] = self._fill_coefs(i)

        # Create a kernel composing all dimensions coefficients
        kernel = np.zeros([np.ptp(ids) + 1 if ids else 1 for ids in stencil_ids])
        for i, ax in enumerate(self.axis):
            slices = tuple(
                [slice(center[j], center[j] + 1) if j != ax else slice(None) for j in range(len(self.arg_shape))]
            )
            shape = [1 if j != ax else kernel.shape[ax] for j in range(len(self.arg_shape))]
            kernel[slices] += stencil_coefs[ax].reshape(shape)

        return kernel, center

    def _fill_coefs(self, i: int) -> typ.Tuple[list, pyct.NDArray, int]:
        r"""
        Defines kernel elements.
        """
        raise NotImplementedError


class _FiniteDifference(_Differential):
    # @Matthieu @Sepand should we make this class available?
    r"""
    Finite Difference base operator. This class is used by :py:class:`~pycsou.operator.linop.diff.PartialDerivative`,
    :py:class:`~pycsou.operator.linop.diff.Gradient` and :py:class:`~pycsou.operator.linop.diff.Hessian`.

    Notes
    -----
    This operator approximates the derivative by finite differences.
    Using the `Finite Difference Coefficients Calculator <https://web.media.mit.edu/~crtaylor/calculator.html>`_
    to construct finite difference approximations for a given a desired derivative order, desired approximation
    accuracy, and finite difference type. Three basic types are considered here:
    - Forward (uses the evaluation point and values to its left),
    - Backward (uses the evaluation point and values to its right), and
    - Central (the evaluation point and values at both sides) finite differences.

    For a given arbitrary order `d\in\mathbb{Z}_{>0}` and accuracy `a\in\mathbb{Z}_{>0}`, the number of indices `s` for
    finite difference is obtained as follows [see
    `ref <https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf>`_]:
    - For central differences --> :math:`s = 2 \lfloor\frac{d + 1}{2}\rfloor - 1 + a`
    - For forward or backward differences --> :math:`s = d + a`

    For a given arbitrary stencil points `s` of length `N` with the order of derivatives `d<N`, the finite difference
    coefficients is obtained by solving the linear equations
    [see `ref <https://web.media.mit.edu/~crtaylor/calculator.html>`_]:

    .. math::
        \left(\begin{array}{ccc}
        s_{1}^{0} & \cdots & s_{N}^{0} \\
        \vdots & \ddots & \vdots \\
        s_{1}^{N-1} & \cdots & s_{N}^{N-1}
        \end{array}\right)\left(\begin{array}{c}
        a_{1} \\
        \vdots \\
        a_{N}
        \end{array}\right)=d t\left(\begin{array}{c}
        \delta_{v_{, d}} \\
        \vdots \\
        \delta_{i, d} \\
        \vdots \\
        \delta_{N-1, d}
        \end{array}\right)

    This class inherits its methods from :py:class:`~pycsou.operator.linop.base.Stencil`. The user is encouraged to read
    the documentation of the :py:class:`~pycsou.operator.linop.base.Stencil` class for a description of accepted
    `kwargs`.

    Remark 1
    --------
    The stencil kernels created can consist on the simultaneous finite differences in different dimensions. For example,
    if `order` is a tuple (1, 1), and `diff_type` is `central`, the following kernel will be created:
    .. math::
        \left(\begin{array}{ccc}
        0 & -0.5 & 0 \\
        -0.5 & 0 & 0.5 \\
        0 & 0.5 & 0
        \end{array}\right)

    Note that this corresponds to the sum of first order partial derivatives:
    .. math::
        \frac{ \partial \mathbf{f} }{\partial x_{0}} + \frac{ \partial \mathbf{f} }{\partial x_{1}}
    And NOT to the second order partial derivative:
    .. math::
        \frac{\partial^{2} \mathbf{f}}{\partial x_{0} \partial x_{1}}
    For the latter kind, :py:class:`~pycsou.operator.linop.diff.PartialDerivative` is the appropriate class.

    Remark 2
    --------
    If `order` is a tuple and different arguments (`diff_type`, `accuracy` and `kwargs`) can be specified for each
    dimension/axis with tuples.

    Examples
    --------
    >>> import numpy as np
    >>> from pycsou.operator.linop.diff import _FiniteDifference
    >>> arg_shape = (3, 3) # Shape of our image
    >>> nsamples = 2 # Number of samples
    >>> order = (1, 1) # Compute derivative of order 1 in first dimension, order 2 in second dimension
    >>> diff = _FiniteDifference(order=order, arg_shape=arg_shape, diff_type="central")
    >>> image = np.ones((nsamples,*arg_shape))
    >>> out = diff(image)
    >>> print(out[0])
    [[ 1.   0.5  0. ]
     [ 0.5  0.  -0.5]
     [ 0.  -0.5 -1. ]]


    See Also
    --------
    :py:class:`~pycsou.operator.linop.diff._Differential`, :py:class:`~pycsou.operator.linop.diff._GaussianDerivative`,
    :py:class:`~pycsou.operator.linop.diff.PartialDerivative`, :py:class:`~pycsou.operator.linop.diff.Gradient`,
    :py:class:`~pycsou.operator.linop.diff.Hessian`.
    """

    def __init__(
        self,
        order: typ.Union[int, tuple[int, ...]],
        arg_shape: pyct.Shape,
        diff_type: typ.Union[str, tuple[str, ...]] = "forward",
        axis: typ.Union[int, tuple[int, ...], None] = None,
        accuracy: typ.Union[int, tuple[int, ...]] = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        order: int | tuple
            Derivative order. If a single integer value is provided, then `axis` should be provided to indicate which
            dimension should be used for differentiation. If a tuple is provided, it should contain as many elements as
            number of dimensions in `axis`.
        arg_shape: tuple
            Shape of the input array
        diff_type: str, tuple
            Type of finite differences ["forward", "backward", "central"]. Defaults to "forward".
        axis: int | tuple | None
            Axis to which apply the derivative. It maps the argument `order` to the specified dimensions of the input
            array. Defaults to None, assuming that the `order` argument has as many elements as dimensions of the input.
        accuracy: int, tuple
            Approximation accuracy to the derivative. See `Notes`.
        kwargs:
            Arguments related to the padding and stencil parameters (see
            :py:class:`~pycsou.operator.linop.base.Stencil`) for more information on both types of parameters.
        """

        self.arg_shape, self.axis = arg_shape, axis
        self._check_inputs(order, param1=diff_type, param1_name="diff_typ", param2=accuracy, param2_name="accuracy")
        kernel, center = self._create_kernel()

        super(_FiniteDifference, self).__init__(kernel=kernel, center=center, arg_shape=arg_shape, **kwargs)

    def _fill_coefs(self, i: int) -> typ.Tuple[list, pyct.NDArray, int]:
        r"""
        Defines kernel elements.
        """
        stencil_ids = self._compute_ids(order=self.order[i], diff_type=self._param1[i], accuracy=self._param2[i])
        stencil_coefs = self._compute_coefficients(stencil_ids=stencil_ids, order=self.order[i])
        center = stencil_ids.index(0)
        return stencil_ids, stencil_coefs, center

    @staticmethod
    def _compute_ids(order: int, diff_type: str, accuracy: float) -> list:
        """
        Computes the Finite difference indices according to the order, type and accuracy.
        """
        if diff_type == "central":
            n_coefs = 2 * ((order + 1) // 2) - 1 + accuracy
            ids = np.arange(-n_coefs // 2, n_coefs // 2 + 1, dtype=int)
        else:
            n_coefs = order + accuracy
            if diff_type == "forward":
                ids = np.arange(0, n_coefs, dtype=int)
            elif diff_type == "backward":
                ids = np.arange(-n_coefs + 1, 1, dtype=int)
            else:
                raise ValueError(
                    f"Incorrect value for variable 'type'. 'type' should be ['forward', 'backward', "
                    f"'central'], but got {diff_type}."
                )
        return ids.tolist()

    @staticmethod
    def _compute_coefficients(stencil_ids: list, order: int) -> pyct.NDArray:
        """
        Computes the finite difference coefficients based on the order and indices.
        """
        # vander doesn't allow precision specification
        stencil_mat = np.vander(
            np.array(stencil_ids),
            increasing=True,
        ).T.astype(pycrt.getPrecision().value)
        vec = np.zeros(len(stencil_ids), dtype=pycrt.getPrecision().value)
        vec[order] = math.factorial(order)
        coefs = np.linalg.solve(stencil_mat, vec)
        return coefs


class _GaussianDerivative(_Differential):
    # @Matthieu @Sepand should we make this class available?
    r"""
    Gaussian derivative operator. This class is used by :py:class:`~pycsou.operator.linop.diff.PartialDerivative`,
    :py:class:`~pycsou.operator.linop.diff.Gradient` and :py:class:`~pycsou.operator.linop.diff.Hessian`.

    Notes
    -----
    This operator approximates the derivative via a Gaussian finite derivative. Computing the derivative of a function
    convolved with a Gaussian is equivalent to convolving the image with the derivative of a Gaussian:
    .. math:
        \frac{\partial}{\partial x}\left[ f(x) * g(x) \right] &= \frac{\partial}{\partial x} * f(x) * g(x) \\
        &= f(x) * \frac{\partial}{\partial x} * g(x) \\
        &= f(x) * \left[\frac{\partial}{\partial x} * f(x) * g(x) \right]

    And because we can compute the derivative of the Gaussian analytically, we can sample it and make a filter out of
    it. This means that we can compute the `exact derivative` of a smoothed signal. It is a different approximation to
    the true derivative of the signal, in contrast to the Finite Difference Method
    (see :py:class:`~pycsou.operator.linop.diff._FiniteDifference`).

    This class inherits its methods from :py:class:`~pycsou.operator.linop.base.Stencil`. The user is encouraged to read
    the documentation of the :py:class:`~pycsou.operator.linop.base.Stencil` class for a description of accepted
    `kwargs`.

    Remark 1
    --------
    The stencil kernels created can consist on the sum of Gaussian Derivatives in different dimensions. For example,
    if `order` is a tuple (1, 1), `sigma` is `1.0` and `truncate` is `1.0`, the following kernel will be created:
    .. math::
        \left(\begin{array}{ccc}
        0 & -0.274 & 0 \\
        -0.274 & 0 & 0.274 \\
        0 & 0.274 & 0
        \end{array}\right)

    Note that this corresponds to the sum of first order partial derivatives:
    .. math::
        \frac{ \partial \mathbf{f} }{\partial x_{0}} + \frac{ \partial \mathbf{f} }{\partial x_{1}}
    And NOT to the second order partial derivative:
    .. math::
        \frac{\partial^{2} \mathbf{f}}{\partial x_{0} \partial x_{1}}
    For the latter kind, :py:class:`~pycsou.operator.linop.diff.PartialDerivative` is the appropriate class.

    Remark 2
    --------
    If `order` is a tuple and different arguments (`diff_type`, `accuracy` and `kwargs`) can be specified for each
    dimension/axis with tuples.

    Examples
    --------
    >>> import numpy as np
    >>> from pycsou.operator.linop.diff import _GaussianDerivative
    >>> arg_shape = (3, 3) # Shape of our image
    >>> nsamples = 2 # Number of samples
    >>> order = (1, 1) # Compute derivative of order 1 in first dimension, order 2 in second dimension
    >>> diff = _GaussianDerivative(order=order, arg_shape=arg_shape, sigma=1., truncate=1.)
    >>> print(diff.stencil_coefs)
    [[ 0.         -0.27406862  0.        ]
     [-0.27406862  0.          0.27406862]
     [ 0.          0.27406862  0.        ]]
    >>> image = np.ones((nsamples,*arg_shape))
    >>> out = diff(image)
    >>> print(out[0])
    [[ 0.54813724  0.27406862  0.        ]
     [ 0.27406862  0.         -0.27406862]
     [ 0.         -0.27406862 -0.54813724]]

    See Also
    --------
    :py:class:`~pycsou.operator.linop.diff._Differential`, :py:class:`~pycsou.operator.linop.diff._FiniteDifference`,
    :py:class:`~pycsou.operator.linop.diff.PartialDerivative`, :py:class:`~pycsou.operator.linop.diff.Gradient`,
    :py:class:`~pycsou.operator.linop.diff.Hessian`.
    """

    def __init__(
        self,
        order: typ.Union[int, tuple[int, ...]],
        arg_shape: pyct.Shape,
        sigma: typ.Union[float, tuple[float, ...]],
        axis: typ.Union[int, tuple[int, ...], None] = None,
        truncate: typ.Union[float, tuple[float, ...]] = 3.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        order: int | tuple
            Derivative order. If a single integer value is provided, then `axis` should be provided to indicate which
            dimension should be used for differentiation. If a tuple is provided, it should contain as many elements as
            number of dimensions in `axis`.
        arg_shape: tuple
            Shape of the input array
        sigma: float | tuple
            Standard deviation of the Gaussian kernel.
        axis: int | tuple | None
            Axis to which apply the derivative. It maps the argument `order` to the specified dimensions of the input
            array. Defaults to None, assuming that the `order` argument has as many elements as dimensions of the input.
        truncate: float | tuple
            Truncate the filter at this many standard deviations.
            Defaults to 3.0.
        kwargs:
            Arguments related to the padding and stencil parameters (see
            :py:class:`~pycsou.operator.linop.base.Stencil`) for more information on both types of parameters.
        """
        self.arg_shape, self.axis = arg_shape, axis
        self._check_inputs(order, param1=sigma, param1_name="sigma", param2=truncate, param2_name="truncate")
        kernel, center = self._create_kernel()
        super(_GaussianDerivative, self).__init__(kernel=kernel, center=center, arg_shape=arg_shape, **kwargs)

    def _fill_coefs(self, i: int) -> typ.Tuple[list, pyct.NDArray, int]:
        r"""
        Defines kernel elements.
        """
        # make the radius of the filter equal to `truncate` standard deviations
        radius = int(self._param2[i] * float(self._param1[i]) + 0.5)
        stencil_coefs = self._gaussian_kernel1d(sigma=self._param1[i], order=self.order[i], radius=radius)
        stencil_ids = [i for i in range(-radius, radius + 1)]
        return stencil_ids, stencil_coefs, radius

    @staticmethod
    def _gaussian_kernel1d(sigma, order: int, radius: int) -> pyct.NDArray:
        """
        Computes a 1-D Gaussian convolution kernel.
        Wraps scipy.ndimage.filters._gaussian_kernel1d
        It flips the output because the original kernel is meant for convolution instead of correlation.
        """
        return np.flip(scif._gaussian_kernel1d(sigma, order, radius))


class _DerivativeOperator(pyco.LinOp):
    """
    Base helper class for Partial Derivative, Gradient and Hessian.
    It defines the common methods for managing input arguments.
    """

    @classmethod
    def _check_directions(cls, arg_shape, param, param_name) -> tuple[int, ...]:
        if not isinstance(param, cabc.Sequence):
            param = (param,)
        # param = tuple(map(int, param))
        else:
            if isinstance(param[0], cabc.Sequence) and (len(param) == 1):
                param = param[0]
        assert (len(param) == 1) | (len(param) == len(arg_shape)), (
            f"The length of {param_name} cannot be larger than the"
            f"number of dimensions ({len(arg_shape)}) defined by `arg_shape`"
        )
        return param

    @classmethod
    def _check_input(
        cls, param, param_name, dtype, directions, directions_name="directions"
    ) -> tuple[typ.Union[int, float, str], ...]:
        if len(directions) == 1:
            if isinstance(directions[0], cabc.Sequence):
                directions = directions[0]
        if not isinstance(param, cabc.Sequence) or isinstance(param, str):
            param = (param,)
        param = tuple(map(dtype, param))

        if dtype in [float, int]:
            assert all(_ > 0.0 for _ in param), f"{param_name} must be strictly positive"

        assert (len(param) == 1) | (len(param) == len(directions)), (
            f"The length of {param_name} cannot be different "
            f"from one or from the number of {directions_name}. Got "
            f"{param_name} with length {len(param)} and "
            f"{len(directions)} {directions_name}."
        )
        if (len(param) == 1) and (len(directions) > 1):
            param = tuple((param[0] for _ in range(len(directions))))

        return param

    @classmethod
    def _check_kwargs(cls, kwargs, arg_shape) -> tuple[dict, ...]:
        # kwargs for each dimension in arg_shape
        if len(arg_shape) == 1:
            if isinstance(arg_shape[0], cabc.Sequence):
                arg_shape = arg_shape[0]

        if kwargs is not None:
            if not isinstance(kwargs, cabc.Sequence):
                kwargs = (kwargs,)
            assert (len(kwargs) == 1) | (len(kwargs) == len(arg_shape)), (
                f"The length of `kwargs` cannot be different from one"
                f" or from the number of dimensions in arg_shape. Got `kwargs` with "
                f"length {len(kwargs)} and {len(arg_shape)} dimensions."
            )
            if (len(kwargs) == 1) and (len(arg_shape) > 1):
                kwargs = tuple((kwargs[0] for _ in range(len(arg_shape))))
        else:
            kwargs = tuple((dict() for _ in range(len(arg_shape))))
        return kwargs


class PartialDerivative(_DerivativeOperator):
    r"""
    Partial derivative operator based on `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.

    This operator computes the partial derivative of a D-dimensional signal:

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}},

    for a given set of directions:

    .. math::

        \mathbf{x}_{i},  \quad i \in [0, \dots, D-1],

    and a given set of derivative orders:

    .. math::

        k_{i},  \quad i \in [0, \dots, D-1],

    with :math:`\quad k = \sum_{i = 0}^{D-1} k_{i}\quad`, i.e.,

    .. math::

        \frac{\partial^{k} \mathbf{f}}{\partial x_{0}^{k_{0}} \, \cdots  \, \partial x_{D-1}^{k_{D-1}}}

    The partial derivative can be approximated by the `finite difference method <https://en.wikipedia.org/wiki/Finite_difference>`_ via the
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` constructor or by the `Gaussian derivative <https://www.crisluengo.net/archives/22/>`_ via
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative` constructor.

    Examples
    --------
    .. code-block:: python3
        import numpy as np
        from pycsou.operator.linop.diff import PartialDerivative

        # Define input
        arg_shape = (3, 3) # Shape of our image
        nsamples = 2 # Number of samples
        image = np.ones((nsamples,*arg_shape)).reshape(nsamples, -1)

        # Specify derivative order at each direction
        df_dx = (1, 0) # Compute derivative of order 1 in first dimension
        d2f_dy2 = (0, 2) # Compute derivative of order 2 in second dimension
        d3f_dxdy2 = (1, 2) # Compute derivative of order 1 in first dimension and der. of order 2 in second dimension

        # Instantiate derivative operators
        diff1 = PartialDerivative.finite_difference(order=df_dx, arg_shape=arg_shape, diff_type="central")
        diff2 = PartialDerivative.finite_difference(order=d2f_dy2, arg_shape=arg_shape, diff_type="central")
        diff3 = PartialDerivative.finite_difference(order=d3f_dxdy2, arg_shape=arg_shape, diff_type="central")

        # Compute derivatives
        out1 = diff1(image).reshape(nsamples, *arg_shape)
        out2 = diff2(image).reshape(nsamples, *arg_shape)
        out3 = diff3(image).reshape(nsamples, *arg_shape)

        # Test
        assert np.allclose(out3, (out1 * out2))
        assert np.allclose(out3, diff1(out2))

    See Also
    --------
    :py:class:`~pycsou.operator.linop.diff._Differential`, :py:class:`~pycsou.operator.linop.diff._FiniteDifference`,
    :py:class:`~pycsou.operator.linop.diff._GaussianDerivative`, :py:class:`~pycsou.operator.linop.diff.Gradient`,
    :py:class:`~pycsou.operator.linop.diff.Hessian`.
    """

    @staticmethod
    def finite_difference(
        arg_shape: pyct.NonAgnosticShape,
        order: tuple[int, ...],
        diff_type: str = "forward",
        accuracy: typ.Union[int, tuple[int, ...]] = 1,
        kwargs: typ.Optional[typ.Union[dict, tuple[dict, ...]]] = None,
    ) -> pyco.LinOp:
        r"""
        Compute the partial derivatives using :py:class:`~pycsou.operator.linop.diff._FiniteDifference`.

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array
        order: tuple
            Derivative order for each dimension. The total order of the partial derivative is the sum
            of elements in the tuple.
        diff_type: str | tuple
            Type of finite differences ['forward, 'backward, 'central']. Defaults to 'forward'. If a string is provided,
            the same `diff_type` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `order`.
        accuracy: float | tuple
            Approximation accuracy to the derivative. See `notes` of :py:class:`~pycsou.operator.linop.diff._FiniteDifference`.
            If a float is provided, the same `accuracy` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `order`.
        kwargs:
            Extra arguments to control the padding and stencil parameters (see
            :py:class:`~pycsou.operator.linop.base.Stencil`) for more information on both types of parameters.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            Partial derivative
        """

        order = PartialDerivative._check_directions(arg_shape, order, "order")
        kwargs = PartialDerivative._check_kwargs(kwargs, order)
        accuracy = PartialDerivative._check_input(accuracy, "accuracy", float, order)
        diff_type = PartialDerivative._check_input(diff_type, "diff_type", str, order)
        fd = pycsou.operator.linop.base.IdentityOp(np.prod(arg_shape))
        for i in range(len(order)):
            if order[i] > 0:
                fd *= _FiniteDifference(
                    order=order[i],
                    arg_shape=arg_shape,
                    diff_type=diff_type[i],
                    axis=i,
                    accuracy=accuracy[i],
                    **kwargs[i],
                )
        return fd

    @staticmethod
    def gaussian_derivative(
        arg_shape: pyct.Shape,
        order: tuple[int, ...],
        sigma: typ.Union[float, tuple[float, ...]] = 1.0,
        truncate: typ.Union[float, tuple[float, ...]] = 3.0,
        kwargs: typ.Optional[typ.Union[dict, tuple[dict, ...]]] = None,
    ) -> pyco.LinOp:
        """
        Compute the partial derivatives using :py:class:`~pycsou.operator.linop.diff._GaussianDerivative`.

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array
        order: tuple
            Derivative order for each dimension. The total order of the partial derivative is the sum
            of elements in the tuple.
        sigma: float | tuple
            Standard deviation for the Gaussian kernel. Defaults to 1.0.
            If a float is provided, the same `sigma` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `order`.
        truncate: float | tuple
            Truncate the filter at this many standard deviations. Defaults to 3.0.
            If a float is provided, the same `truncate` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `order`.
        kwargs:
            Extra arguments to control the padding and stencil parameters (see
            :py:class:`~pycsou.operator.linop.base.Stencil`) for more information on both types of parameters.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            Partial derivative
        """

        order = PartialDerivative._check_directions(arg_shape, order, "order")
        kwargs = PartialDerivative._check_kwargs(kwargs, order)
        sigma = PartialDerivative._check_input(sigma, "sigma", float, order)
        truncate = PartialDerivative._check_input(truncate, "truncate", float, order)
        gd = pycsou.operator.linop.base.IdentityOp(np.prod(arg_shape))
        for i in range(len(order)):
            if order[i] > 0:
                gd *= _GaussianDerivative(
                    order=order[i],
                    arg_shape=arg_shape,
                    sigma=sigma[i],
                    axis=i,
                    truncate=truncate[i],
                    **kwargs[i],
                )
        return gd


class _StackPartialDerivatives(_DerivativeOperator):
    r"""
    Helper class for Gradient and Hessian.

    Defines a method for computing and stacking partial derivatives.

    See Also
    --------
    :py:class:`~pycsou.operator.linop.diff.Gradient`,
    :py:class:`~pycsou.operator.linop.diff.Hessian`.
    """

    @classmethod
    def _finite_difference(
        cls,
        arg_shape: pyct.NonAgnosticShape,
        order: typ.Union[typ.Tuple[int, ...], typ.Tuple[typ.Tuple[int, ...], ...], int],
        directions: typ.Optional[typ.Union[int, tuple[int, ...]]] = None,
        diff_type: typ.Union[str, tuple[str, ...]] = "forward",
        accuracy: typ.Union[int, tuple[int, ...]] = 1,
        kwargs: typ.Optional[typ.Union[dict, tuple[dict]]] = None,
    ) -> pyco.LinOp:
        """
        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array
        order: int
            Order of the partial derivatives
        directions: int, tuple, None
            Gradient directions. Defaults to `None`, which computes the gradient for all directions.
        diff_type: str
            Type of finite differences ["forward", "backward", "central"]. Defaults to "forward".
        accuracy:
            Approximation accuracy to the derivative. See `Notes` of
            :py:class:`~pycsou.operator.linop.diff._FiniteDifference`.
        kwargs:
            Extra arguments to control the padding and stencil parameters (see
            :py:class:`~pycsou.operator.linop.base.Stencil`) for more information on both types of parameters.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            Stack of partial derivatives
        """

        order = cls._check_input(order, "order", int, directions)
        return cls._stack_diff_ops(
            arg_shape=arg_shape,
            directions=directions,
            diff_method="fd",
            order=order,
            param1=diff_type,
            param2=accuracy,
            kwargs=kwargs,
        )

    @classmethod
    def _gaussian_derivative(
        cls,
        arg_shape: pyct.NonAgnosticShape,
        order: typ.Union[typ.Tuple[int, ...], typ.Tuple[typ.Tuple[int, ...], ...], int],
        directions: typ.Optional[typ.Union[int, tuple[int, ...]]] = None,
        sigma: typ.Union[float, tuple[float, ...]] = 1.0,
        truncate: typ.Union[float, tuple[float, ...]] = 3.0,
        kwargs: typ.Optional[typ.Union[dict, tuple[dict]]] = None,
    ) -> pyco.LinOp:
        """
        Compute the gradient using :py:class:`~pycsou.operator.linop.diff._GaussianDerivative`.
        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array
        directions: int, tuple, None
            Gradient directions. Defaults to `None`, which computes the gradient for all directions.
        sigma: float | tuple
            Standard deviation for the Gaussian kernel. Defaults to 1.0.
        truncate: float | tuple
            Truncate the filter at this many standard deviations. Defaults to 3.0.
        kwargs:
            Extra arguments to control the padding and stencil parameters (see
            :py:class:`~pycsou.operator.linop.base.Stencil`) for more information on both types of parameters.

        Returns
        -------
        gradient: NDArray
        """
        order = cls._check_input(order, "order", int, directions)
        return cls._stack_diff_ops(
            arg_shape=arg_shape,
            directions=directions,
            diff_method="gd",
            order=order,
            param1=sigma,
            param2=truncate,
            kwargs=kwargs,
        )

    @staticmethod
    def _stack_diff_ops(arg_shape, directions, diff_method, order, param1, param2, kwargs):

        dif_op = []
        for i in range(0, len(directions)):
            _order = np.zeros_like(arg_shape)
            _order[directions[i]] = order[i]
            if diff_method == "fd":
                dif_op.append(
                    PartialDerivative.finite_difference(
                        arg_shape=arg_shape,
                        order=tuple(_order),
                        kwargs=kwargs[i],
                        diff_type=param1[i],
                        accuracy=param2[i],
                    )
                )
            elif diff_method == "gd":
                dif_op.append(
                    PartialDerivative.gaussian_derivative(
                        arg_shape=arg_shape,
                        order=tuple(_order),
                        kwargs=kwargs[i],
                        sigma=param1[i],
                        truncate=param2[i],
                    )
                )
        return pycc.vstack(dif_op)


class Gradient(_StackPartialDerivatives):
    r"""
    Gradient Operator based on `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.

    This operator computes the first order partial derivatives of a D-dimensional signal:

    .. math::

        \mathbf{f} \in \mathbb{R}^{N_{0}, \dots, N_{D-1}},

    for the desired directions :math:`\mathbf{x}_{i}` for :math:`i \in [0, \dots, D]`, i.e.,

    .. math::

        \nabla \mathbf{f} = \left( \frac{ \partial \mathbf{f} }{ \partial \mathbf{x}_{1} }, \quad \ldots \quad, \frac{ \partial \mathbf{f} }{ \partial \mathbf{x}_{n} } \right).

    The gradient can be approximated by the `finite difference method <https://en.wikipedia.org/wiki/Finite_difference>`_ via the
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` constructor or by the `Gaussian derivative <https://www.crisluengo.net/archives/22/>`_ via
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative` constructor.

    Examples
    --------
    .. code-block:: python3
        import numpy as np
        from pycsou.operator.linop.diff import Gradient

        # Define input image
        arg_shape = (3, 3)
        nsamples = 2
        image = np.ones((nsamples,*arg_shape)).reshape(nsamples, -1)

        # Instantiate gradient operator
        grad = Gradient.gaussian_derivative(arg_shape=arg_shape, sigma=1.0)

        # Compute gradient
        out = grad(image)

    See Also
    --------
    :py:class:`~pycsou.operator.linop.diff._Differential`, :py:class:`~pycsou.operator.linop.diff._FiniteDifference`,
    :py:class:`~pycsou.operator.linop.diff._GaussianDerivative`,
    :py:class:`~pycsou.operator.linop.diff.PartialDerivative`, :py:class:`~pycsou.operator.linop.diff.Hessian`.
    """

    @staticmethod
    def finite_difference(
        arg_shape: pyct.NonAgnosticShape,
        directions: typ.Optional[typ.Union[int, tuple[int, ...]]] = None,
        diff_type: str = "forward",
        accuracy: typ.Union[int, tuple[int, ...]] = 1,
        kwargs: typ.Optional[typ.Union[dict, tuple[dict]]] = None,
    ) -> pyco.LinOp:
        """
        Compute the gradient using :py:class:`~pycsou.operator.linop.diff._FiniteDifference`.

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array
        directions: int, tuple, None
            Gradient directions. Defaults to `None`, which computes the gradient for all directions.
        diff_type: str | tuple
            Type of finite differences ['forward, 'backward, 'central']. Defaults to 'forward'. If a string is provided,
            the same `diff_type` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `directions`.
        accuracy: float | tuple
            Approximation accuracy to the derivative. See `notes` of :py:class:`~pycsou.operator.linop.diff._FiniteDifference`.
            If a float is provided, the same `accuracy` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `directions`.
        kwargs:
            Extra arguments to control the padding and stencil parameters (see
            :py:class:`~pycsou.operator.linop.base.Stencil`) for more information on both types of parameters.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            Gradient
        """
        directions = tuple([i for i in range(len(arg_shape))]) if directions is None else directions
        directions = Gradient._check_directions(arg_shape, directions, "directions")
        kwargs = Gradient._check_kwargs(kwargs, directions)
        accuracy = Gradient._check_input(accuracy, "accuracy", float, directions)
        diff_type = Gradient._check_input(diff_type, "diff_type", str, directions)
        return Gradient._finite_difference(arg_shape, 1, directions, diff_type, accuracy, kwargs)

    @staticmethod
    def gaussian_derivative(
        arg_shape: pyct.NonAgnosticShape,
        directions: typ.Optional[typ.Union[int, tuple[int, ...]]] = None,
        sigma: typ.Union[float, tuple[float, ...]] = 1.0,
        truncate: typ.Union[float, tuple[float, ...]] = 3.0,
        kwargs: typ.Optional[typ.Union[dict, tuple[dict]]] = None,
    ) -> pyco.LinOp:
        """
        Compute the gradient using :py:class:`~pycsou.operator.linop.diff._GaussianDerivative`.

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array
        directions: int, tuple, None
            Gradient directions. Defaults to `None`, which computes the gradient for all directions.
        sigma: float | tuple
            Standard deviation for the Gaussian kernel. Defaults to 1.0.
            If a float is provided, the same `sigma` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `directions`.
        truncate: float | tuple
            Truncate the filter at this many standard deviations. Defaults to 3.0.
            If a float is provided, the same `truncate` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `directions`.
        kwargs:
            Extra arguments to control the padding and stencil parameters (see
            :py:class:`~pycsou.operator.linop.base.Stencil`) for more information on both types of parameters.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            Gradient
        """
        directions = tuple([i for i in range(len(arg_shape))]) if directions is None else directions
        directions = Gradient._check_directions(arg_shape, directions, "directions")
        kwargs = Gradient._check_kwargs(kwargs, directions)
        sigma = Gradient._check_input(sigma, "sigma", float, directions)
        truncate = Gradient._check_input(truncate, "truncate", float, directions)
        return Gradient._gaussian_derivative(arg_shape, 1, directions, sigma, truncate, kwargs)


class Hessian(_StackPartialDerivatives):
    r"""
    Hessian Operator based on  `Numba stencils <https://numba.pydata.org/numba-doc/latest/user/stencil.html>`_.

    The Hessian matrix or Hessian is a square matrix of second-order partial derivatives:

    .. math::

        \mathbf{H}_{f} = \begin{bmatrix}
        \dfrac{ \partial^{2}\mathbf{f} }{ \partial \mathbf{x}_{1}^{2} } &  \dfrac{ \partial^{2}\mathbf{f} }{ \partial \mathbf{x}_{1}\,\partial \mathbf{x}_{2} } & \cdots & \dfrac{ \partial^{2}\mathbf{f} }{ \partial \mathbf{x}_{1} \, \partial \mathbf{x}_{D} } \\
        \dfrac{ \partial^{2}\mathbf{f} }{ \partial \mathbf{x}_{2} \, \partial \mathbf{x}_{1} } & \dfrac{ \partial^{2}\mathbf{f} }{ \partial \mathbf{x}_{2}^{2} } & \cdots & \dfrac{ \partial^{2}\mathbf{f} }{\partial \mathbf{x}_{2} \,\partial \mathbf{x}_{D}} \\
        \vdots & \vdots & \ddots & \vdots \\
        \dfrac{ \partial^{2}\mathbf{f} }{ \partial \mathbf{x}_{D} \, \partial \mathbf{x}_{1} } & \dfrac{ \partial^{2}\mathbf{f} }{ \partial \mathbf{x}_{n} \, \partial \mathbf{x}_{2} } & \cdots & \dfrac{ \partial^{2}\mathbf{f} }{ \partial \mathbf{x}_{D}^{2}}
        \end{bmatrix}

    The Hessian can be approximated by the `finite difference method <https://en.wikipedia.org/wiki/Finite_difference>`_ via the
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.finite_difference` constructor or by the `Gaussian derivative <https://www.crisluengo.net/archives/22/>`_ via
    :py:meth:`~pycsou.operator.linop.diff.PartialDerivative.gaussian_derivative` constructor.

    Notes
    -----

    Due to the (possibly) large size of the full Hessian, four different options are handled:

    * [mode 0] ``directions`` is an integer (e.g., ``directions=0`` :math:`\rightarrow \partial^{2}\mathbf{f}/\partial x_{0}^{2},)`.
    * [mode 1] ``directions`` is tuple of length 2 (e.g., ``directions=(0,1)`` :math:`\rightarrow  \partial^{2}\mathbf{f}/\partial x_{0}\partial x_{1},)`.
    * [mode 2]  ``directions`` is tuple of tuples (e.g., ``directions=((0,0), (0,1))`` :math:`\rightarrow  \left(\frac{ \partial^{2}\mathbf{f} }{ \partial x_{0}^{2} }, \frac{ \partial^{2}\mathbf{f} }{ \partial x_{0}\partial x_{1} }\right)`.
    * [mode 3] ``directions`` is 'all'  computes the Hessian for all directions, i.e., :math:`\rightarrow  \left(\frac{ \partial^{2}\mathbf{f} }{ \partial x_{0}^{2} }, \frac{ \partial^{2}\mathbf{f} }{ \partial x_{0}\partial x_{1} }, \, \ldots , \, \frac{ \partial^{2}\mathbf{f} }{ \partial x_{D}^{2} }\right)`.

    Remark
    ______

    If the user wants to adjust the padding options, `kwargs` should be a tuple with as a tuple with one
    dictionary or an empty list per `arg_shape` dimensions. If only a dict is provided, equal boundary conditions will
    be used.

    Examples
    --------
    .. code-block:: python3
        import numpy as np
        from pycsou.operator.linop.diff import Hessian, PartialDerivative

        # Define input
        arg_shape = (3, 3)
        nsamples = 2
        image = np.ones((nsamples,*arg_shape)).reshape(nsamples, -1)

        # Instantiate Hessian and derivative operators (for comparison)
        directions = "all"
        hes = Hessian.finite_difference(arg_shape=arg_shape, diff_type="central", directions=directions)
        pd1 = PartialDerivative.finite_difference(order=(2, 0), arg_shape=arg_shape, diff_type="central")
        pd2 = PartialDerivative.finite_difference(order=(1, 1), arg_shape=arg_shape, diff_type="central")
        pd3 = PartialDerivative.finite_difference(order=(0, 2), arg_shape=arg_shape, diff_type="central")

        # Compute derivatives
        out_hess = hes(image).reshape(nsamples, 3, *arg_shape)
        out_pd1 = pd1(image).reshape(nsamples, *arg_shape)
        out_pd2 = pd2(image).reshape(nsamples, *arg_shape)
        out_pd3 = pd3(image).reshape(nsamples, *arg_shape)

        # Test
        assert np.allclose(out_hess[:, 0], out_pd1)
        assert np.allclose(out_hess[:, 1], out_pd2)
        assert np.allclose(out_hess[:, 2], out_pd3)

     See Also
    --------
    :py:class:`~pycsou.operator.linop.diff._Differential`, :py:class:`~pycsou.operator.linop.diff._FiniteDifference`,
    :py:class:`~pycsou.operator.linop.diff._GaussianDerivative`,
    :py:class:`~pycsou.operator.linop.diff.PartialDerivative`, :py:class:`~pycsou.operator.linop.diff.Gradient`.
    """

    @staticmethod
    def finite_difference(
        arg_shape: pyct.NonAgnosticShape,
        directions: typ.Union[str, tuple[int, int], tuple[tuple[int, int], ...]] = "all",
        diff_type: typ.Union[str, tuple[str, ...]] = "forward",
        accuracy: typ.Union[int, tuple[int, ...]] = 1,
        kwargs: typ.Optional[typ.Union[dict, tuple[dict]]] = None,
    ) -> typ.Union[pyco.LinOp, typ.Tuple[pyco.LinOp, ...]]:
        """

        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array
        directions: int, tuple, None
            Hessian directions. Defaults to `all`, which computes the Hessian for all directions.
        diff_type: str | tuple
            Type of finite differences ['forward, 'backward, 'central']. Defaults to 'forward'. If a string is provided,
            the same `diff_type` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `arg_shape`.
        accuracy: float | tuple
            Approximation accuracy to the derivative. See `notes` of :py:class:`~pycsou.operator.linop.diff._FiniteDifference`.
            If a float is provided, the same `accuracy` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `arg_shape`.
        kwargs:
            Extra arguments to control the padding and stencil parameters (see
            :py:class:`~pycsou.operator.linop.base.Stencil`) for more information on both types of parameters.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            Hessian
        """

        directions, order = Hessian._check_directions_and_order(arg_shape, directions)
        diff_type, accuracy, kwargs = Hessian._check_inputs(
            arg_shape=arg_shape,
            kwargs=kwargs,
            param1=diff_type,
            param2=accuracy,
            finite_vs_gaussian="finite",
            ndirs=len(directions),
        )

        return Hessian._finite_difference(
            arg_shape=arg_shape,
            directions=directions,
            order=order,
            diff_type=diff_type,
            accuracy=accuracy,
            kwargs=kwargs,
        )

    @staticmethod
    def gaussian_derivative(
        arg_shape: pyct.NonAgnosticShape,
        directions: typ.Union[str, tuple[int, int], tuple[tuple[int, int], ...]] = "all",
        sigma: typ.Union[float, tuple[float, ...]] = 1.0,
        truncate: typ.Union[float, tuple[float, ...]] = 3.0,
        kwargs: typ.Optional[typ.Union[dict, tuple[dict]]] = None,
    ) -> typ.Union[pyco.LinOp, typ.Tuple[pyco.LinOp, ...]]:
        """
        Parameters
        ----------
        arg_shape: tuple
            Shape of the input array
        directions: int, tuple, None
            Hessian directions. Defaults to `all`, which computes the Hessian for all directions.
        sigma: float | tuple
            Standard deviation for the Gaussian kernel. Defaults to 1.0.
            If a float is provided, the same `sigma` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `arg_shape`.
        truncate: float | tuple
            Truncate the filter at this many standard deviations. Defaults to 3.0.
            If a float is provided, the same `truncate` is assumed for all dimensions. If a tuple is provided, it should have as many elements as `arg_shape`.
        kwargs:
            Extra arguments to control the padding and stencil parameters (see
            :py:class:`~pycsou.operator.linop.base.Stencil`) for more information on both types of parameters.

        Returns
        -------
        op: :py:class:`~pycsou.abc.operator.LinOp`
            Hessian
        """

        directions, order = Hessian._check_directions_and_order(arg_shape, directions)
        sigma, truncate, kwargs = Hessian._check_inputs(
            arg_shape=arg_shape,
            kwargs=kwargs,
            param1=sigma,
            param2=truncate,
            finite_vs_gaussian="gaussian",
            ndirs=len(directions),
        )

        return Hessian._gaussian_derivative(
            arg_shape=arg_shape,
            directions=directions,
            order=order,
            sigma=sigma,
            truncate=truncate,
            kwargs=kwargs,
        )

    @classmethod
    def _check_directions_and_order(
        cls, arg_shape, directions
    ) -> typ.Tuple[typ.Union[tuple[int, ...], tuple[tuple[int, ...], ...]], bool]:
        def _check_directions(_directions):
            assert all(0 <= _ <= (len(arg_shape) - 1) for _ in _directions), (
                "Direction values must be between 0 and " "the number of dimensions in `arg_shape`."
            )

        if not isinstance(directions, cabc.Sequence):
            # This corresponds to [mode 0] in `Notes`
            directions = [directions, directions]
            _check_directions(directions)
            directions = (directions,)
        else:
            if isinstance(directions, str):
                # This corresponds to [mode 3] in `Notes`
                assert directions == "all", (
                    f"Value for `directions` not implemented. The accepted directions types are"
                    f"int, tuple or a str with the value `all`."
                )
                directions = tuple(
                    list(_) for _ in itertools.combinations_with_replacement(np.arange(len(arg_shape)).astype(int), 2)
                )
            elif not isinstance(directions[0], cabc.Sequence):
                # This corresponds to [mode 2] in `Notes`
                assert len(directions) == 2, (
                    "If `directions` is a tuple, it should contain two elements, corresponding "
                    "to the i-th an j-th elements (dx_i and dx_j)"
                )
                directions = list(directions)
                _check_directions(directions)
                directions = (directions,)
            else:
                # This corresponds to [mode 3] in `Notes`
                for direction in directions:
                    _check_directions(direction)

        _directions = [
            list(direction) if (len(np.unique(direction)) == len(direction)) else np.unique(direction).tolist()
            for direction in directions
        ]

        _order = [3 - len(np.unique(direction)) for direction in directions]

        return _directions, _order

    @classmethod
    def _check_inputs(cls, arg_shape, kwargs, param1, param2, finite_vs_gaussian, ndirs):
        param_names = {
            "finite": {"param1": "diff_type", "param2": "accuracy"},
            "gaussian": {"param1": "sigma", "param2": "truncate"},
        }
        kwargs = (Hessian._check_kwargs(kwargs, arg_shape),) * ndirs
        param1 = (Hessian._check_param(param1, param_names[finite_vs_gaussian]["param1"], arg_shape),) * ndirs
        param2 = (Hessian._check_param(param2, param_names[finite_vs_gaussian]["param2"], arg_shape),) * ndirs
        return param1, param2, kwargs

    @classmethod
    def _check_param(cls, param, param_name, arg_shape):
        # kwargs for each dimension in arg_shape
        if param is not None:
            if not isinstance(param, cabc.Sequence) or isinstance(param, str):
                param = (param,)
            assert (len(param) == 1) | (len(param) == len(arg_shape)), (
                f"The length of `kwargs` cannot be different from one"
                f" or from the number of dimensions in arg_shape. Got `{param_name}` with "
                f"length {len(param)} and {len(arg_shape)} dimensions."
            )
            if (len(param) == 1) and (len(arg_shape) > 1):
                param = tuple((param[0] for _ in range(len(arg_shape))))
        else:
            param = tuple((dict() for _ in range(len(arg_shape))))
        return param


def DirectionalDerivative(arg_shape: pyct.Shape, order: int, directions: pyct.NDArray, diff_method="gd", **diff_kwargs):
    r"""
    Directional derivative.
    Computes the first or second directional derivative of a multi-dimensional array along either a single common
    direction for all entries of the array or a different direction for each entry of the array.

    TODO AFTER SEPAND'S PR --> Test with DiagonalOp

    Parameters
    ----------
    arg_shape: tuple
        Shape of the input array
    order: int
        Order of the directional derivative (restricted to 1 or 2).
    directions: NDArray
        Single direction (array of size :math:`n_\text{dims}`) or group of directions
        (array of size :math:`[n_\text{dims} \times n_{d_0} \times ... \times n_{d_{n_\text{dims}}}]`)
    diff_method: str ['gd', 'fd']
        Method used to approximate the derivative. It can be the finite difference method (`fd`) or the Gaussian
        derivative (`gd`).
    diff_kwargs:
        Arguments related to the padding and stencil parameters (see
        :py:class:`~pycsou.operator.linop.diff.PartialDerivative`). See
        :py:class:`~pycsou.operator.linop.base.Stencil` for more information on both types of parameters.

    Returns
    -------



    The *first-order* ``DirectionalDerivative`` applies a derivative to a multi-dimensional array along the direction
    defined by the unitary vector :math:`\mathbf{v}`:

    .. math::
        d_\mathbf{v}f =
            \langle\nabla f, \mathbf{v}\rangle,

    or along the directions defined by the unitary vectors

    :math:`\mathbf{v}(x, y)`:
    .. math::
        d_\mathbf{v}(x,y) f =
            \langle\nabla f(x,y), \mathbf{v}(x,y)\rangle

    where we have here considered the 2-dimensional case.
    Note that the 2D case, choosing :math:`\mathbf{v}=[1,0]` or :math:`\mathbf{v}=[0,1]`
    is equivalent to the first-order ``PartialDerivative`` operator applied to axis 0 or 1 respectively.

    The *second-order* ``DirectionalDerivative`` applies a second-order derivative to a multi-dimensional array along
    the direction defined by the unitary vector :math:`\mathbf{v}`:
    .. math::
        d^2_\mathbf{v} f =
            - d_\mathbf{v}^\ast (d_\mathbf{v} f)
    where :math:`d_\mathbf{v}` is the first-order directional derivative implemented by
    :py:func:`~pycsou.operator.linop.diff.FirstDirectionalDerivative`. The above formula generalises the well-known
    relationship:
    .. math::
        \Delta f= -\text{div}(\nabla f),
    where minus the divergence operator is the adjoint of the gradient.

    **Note that problematic values at edges are set to zero.** TODO double check this statement


    See Also
    --------
    :py:func:`~pycsou.operator.linop.diff.Gradient`, :py:func:`~pycsou.operator.linop.diff.DirectionalGradient`
    """

    assert (order == 1) | (order == 2), "`order` must be either 1 or 2"
    order = (order,) * len(arg_shape)

    if diff_method == "fd":
        diff = Gradient.finite_difference(arg_shape=arg_shape, directions=order, **diff_kwargs)
    elif diff_method == "gd":
        diff = Gradient.gaussian_derivative(arg_shape=arg_shape, directions=order, **diff_kwargs)
    else:
        raise NotImplementedError

    if directions.ndim == 1:
        dop = pycob.DiagonalOp(directions) * diff
    else:
        dop = pycob.DiagonalOp(directions.ravel()) * diff

    # Get gradient ( len(arg_shape), nsamples, *arg_shape),
    # Multiply by directions (len(arg_shape), arg_shape)

    if order == 1:
        return dop
    elif order == 2:
        return -dop.adjoint(dop)


def DirectionalGradient(arg_shape: pyct.Shape, diff_method="gd", **diff_kwargs):
    r"""
    Directional gradient.
    Computes the directional derivative of a multi-dimensional array along multiple ``directions`` for each entry of
    the array.

    Parameters
    ----------
    arg_shape
    diff_method
    diff_kwargs

    Returns
    -------

    Notes
    -----
    The ``DirectionalGradient`` of a multivariate function :math:`f(\mathbf{x})` is defined as:
    .. math::
        d_{\mathbf{v}_1(\mathbf{x}),\ldots,\mathbf{v}_N(\mathbf{x})} f =
            \left[\begin{array}{c}
            \langle\nabla f, \mathbf{v}_1(\mathbf{x})\rangle\\
            \vdots\\
            \langle\nabla f, \mathbf{v}_N(\mathbf{x})\rangle
            \end{array}\right],
    where :math:`d_\mathbf{v}` is the first-order directional derivative
    implemented by :py:func:`~pycsou.operator.linop.diff.FirstDirectionalDerivative`.

    See Also
    --------
    :py:func:`~pycsou.operator.linop.diff.Gradient`, :py:func:`~pycsou.operator.linop.diff.DirectionalDerivative`
    """

    dir_deriv = []
    for i in range(len(arg_shape)):
        directions = [0 if j != i else 1 for j in range(len(arg_shape))]
        dir_deriv.append(
            DirectionalDerivative(
                arg_shape=arg_shape, order=1, directions=directions, diff_method=diff_method, **diff_kwargs
            )
        )
    return pycc.vstack(dir_deriv)


class Jacobian(pyco.LinOp):
    """
    NOTES
    _____

    See discussion about approximating a second order derivative from two successive first order derivatives (e.g.,
    approximating the Hessian with the Jacobian of the gradient --> https://math.stackexchange.com/questions/3756717/finite-differences-second-derivative-as-successive-application-of-the-first-deri

    Examples
    --------
    >>> import numpy as np
    >>> from pycsou.operator.linop.diff import Hessian, Jacobian, Gradient
    >>> arg_shape = (4, 4)  # Shape of our image
    >>> nsamples = 2  # Number of samples
    >>> directions = ((0, 0), (0, 1), (1, 0), (1, 1))
    >>> hessian = Hessian.finite_difference(arg_shape=arg_shape, directions=directions, diff_type="central")
    >>> grad = Gradient.finite_difference(arg_shape=arg_shape, diff_type="central")
    >>> jacobian = Jacobian(arg_shape, diff_method="fd",  diff_type="central")
    >>> image = np.ones((nsamples, *arg_shape))
    >>> image[:, [0,-1]] = 0
    >>> image[:, :, [0,-1]] = 0
    >>> print(image[0])
    [[0. 0. 0. 0.]
     [0. 1. 1. 0.]
     [0. 1. 1. 0.]
     [0. 0. 0. 0.]]
    >>> image = image.reshape(nsamples, -1)
    >>> out = hessian(image).reshape(nsamples, len(arg_shape) ** 2, *arg_shape)
    >>> out2 = jacobian(grad(image).reshape(nsamples, len(arg_shape), -1)).reshape(nsamples, len(arg_shape) ** 2, *arg_shape)
    >>> assert np.allclose(out, out2)
    >>> print(out[0])
    [[[ 0.    1.    1.    0.  ]
      [ 0.   -1.   -1.    0.  ]
      [ 0.   -1.   -1.    0.  ]
      [ 0.    1.    1.    0.  ]]
    <BLANKLINE>
     [[ 0.25  0.25 -0.25 -0.25]
      [ 0.25  0.25 -0.25 -0.25]
      [-0.25 -0.25  0.25  0.25]
      [-0.25 -0.25  0.25  0.25]]
    <BLANKLINE>
     [[ 0.25  0.25 -0.25 -0.25]
      [ 0.25  0.25 -0.25 -0.25]
      [-0.25 -0.25  0.25  0.25]
      [-0.25 -0.25  0.25  0.25]]
    <BLANKLINE>
     [[ 0.    0.    0.    0.  ]
      [ 1.   -1.   -1.    1.  ]
      [ 1.   -1.   -1.    1.  ]
      [ 0.    0.    0.    0.  ]]]
    >>> print(out2[0])
    [[[ 0.    0.25  0.25  0.  ]
      [ 0.   -0.5  -0.5   0.  ]
      [ 0.   -0.5  -0.5   0.  ]
      [ 0.    0.25  0.25  0.  ]]
    <BLANKLINE>
     [[ 0.25  0.25 -0.25 -0.25]
      [ 0.25  0.25 -0.25 -0.25]
      [-0.25 -0.25  0.25  0.25]
      [-0.25 -0.25  0.25  0.25]]
    <BLANKLINE>
     [[ 0.25  0.25 -0.25 -0.25]
      [ 0.25  0.25 -0.25 -0.25]
      [-0.25 -0.25  0.25  0.25]
      [-0.25 -0.25  0.25  0.25]]
    <BLANKLINE>
     [[ 0.    0.    0.    0.  ]
      [ 0.25 -0.5  -0.5   0.25]
      [ 0.25 -0.5  -0.5   0.25]
      [ 0.    0.    0.    0.  ]]]
    """

    def __init__(self, arg_shape, diff_method="gd", **kwargs):
        self.arg_shape = arg_shape

        if diff_method == "fd":
            self.grad = Gradient.finite_difference(arg_shape, **kwargs)
        elif diff_method == "gd":
            self.grad = Gradient.gaussian_derivative(arg_shape, **kwargs)

        size_dom = len(arg_shape) * np.prod(arg_shape)
        size_codom = len(arg_shape) * len(arg_shape) * np.prod(arg_shape)
        super(Jacobian, self).__init__((size_codom, size_dom))

    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        nsamples = arr.shape[0]
        # arr = arr.reshape(nsamples, len(self.arg_shape), *self.arg_shape)
        # out = xp.zeros((nsamples, len(self.arg_shape), len(self.arg_shape), *self.arg_shape))
        # out = xp.zeros((nsamples, len(self.arg_shape), *self.arg_shape))
        # for d in range(len(self.arg_shape)):
        # out[:, d] = self.grad(arr[:, d].reshape(nsamples, -1)).reshape(nsamples, len(self.arg_shape), *self.arg_shape)
        return self.grad(arr)
        # return out.reshape(nsamples, -1)
