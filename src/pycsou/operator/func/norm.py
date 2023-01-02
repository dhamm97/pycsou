import numpy as np
import scipy.optimize as sopt

import pycsou.abc as pyca
import pycsou.math.linalg as pylinalg
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "L1Norm",
    "L2Norm",
    "SquaredL2Norm",
    "SquaredL1Norm",
    "LInftyNorm",
]


class ShiftLossMixin:
    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        from pycsou.operator.func.loss import shift_loss

        op = shift_loss(op=self, data=data)
        return op


class L1Norm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    :math:`\ell_1`-norm, :math:`\Vert\mathbf{x}\Vert_1:=\sum_{i=1}^N |x_i|`.
    """

    def __init__(self, dim: pyct.Integer = None):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)

        Notes
        -----
        The operator's Lipschitz constant is set to :math:`\infty` if domain-agnostic.
        It is recommended to set `dim` explicitly to compute a tight closed-form.
        """
        super().__init__(shape=(1, dim))
        if dim is None:
            self._lipschitz = np.inf
        else:
            self._lipschitz = np.sqrt(dim)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, ord=1, axis=-1, keepdims=True)
        return y

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.fmax(0, xp.fabs(arr) - tau)
        y *= xp.sign(arr)
        return y


class L2Norm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    :math:`\ell_2`-norm, :math:`\Vert\mathbf{x}\Vert_2:=\sqrt{\sum_{i=1}^N |x_i|^2}`.
    """

    def __init__(self, dim: pyct.Integer = None):
        super().__init__(shape=(1, dim))
        self._lipschitz = 1
        self._diff_lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, ord=2, axis=-1, keepdims=True)
        return y

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        scale = 1 - tau / xp.fmax(self.apply(arr), tau)

        y = arr.copy()
        y *= scale.astype(dtype=arr.dtype)
        return y


class SquaredL2Norm(pyca._QuadraticFunc):
    r"""
    :math:`\ell^2_2`-norm, :math:`\Vert\mathbf{x}\Vert^2_2:=\sum_{i=1}^N |x_i|^2`.
    """

    def __init__(self, dim: pyct.Integer = None):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        """
        super().__init__(shape=(1, dim))
        self._lipschitz = np.inf
        self._diff_lipschitz = 2

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, axis=-1, keepdims=True)
        y **= 2
        return y

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        return 2 * arr

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        y = arr.copy()
        y /= 2 * tau + 1
        return y

    def _hessian(self) -> pyct.OpT:
        from pycsou.operator.linop import IdentityOp

        if self.dim is None:
            msg = "\n".join(
                [
                    "hessian: domain-agnostic functionals unsupported.",
                    f"Explicitly set `dim` in {self.__class__}.__init__().",
                ]
            )
            raise ValueError(msg)
        return IdentityOp(dim=self.dim).squeeze()


class SquaredL1Norm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    :math:`\ell^2_1`-norm, :math:`\Vert\mathbf{x}\Vert^2_1:=(\sum_{i=1}^N |x_i|)^2`.
    """

    def __init__(self, dim: pyct.Integer = None, prox_algo: str = "sort"):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        prox_algo: str
            Algorithm used for computing the proximal operator:

            * 'root' uses [FirstOrd]_ Lemma 6.70
            * 'sort' uses [OnKerLearn]_ Algorithm 2 (faster).

        Notes
        -----
        :py:meth:`~pycsou.operator.func.norm.SquaredL1Norm.prox` will always use the root method
        when applied on Dask inputs. (Reason: sorting Dask inputs at scale is discouraged.)
        """
        super().__init__(shape=(1, dim))
        self._lipschitz = np.inf

        algo = prox_algo.strip().lower()
        assert algo in ("root", "sort")
        self._algo = algo

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = pylinalg.norm(arr, ord=1, axis=-1, keepdims=True)
        y **= 2
        return y

    def _prox_root(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        if xp.linalg.norm(arr) > 0:
            mu_max = xp.max(xp.fabs(arr) ** 2) / (4 * tau)
            mu_min = 1e-12
            func = lambda mu: xp.sum(xp.fmax(xp.fabs(arr) * xp.sqrt(tau / mu) - 2 * tau, 0)) - 1
            mu_star = sopt.brentq(func, a=mu_min, b=mu_max)
            lambda_ = xp.fmax(xp.abs(arr) * xp.sqrt(tau / mu_star) - 2 * tau, 0)
            lambda_ = lambda_.astype(arr.dtype)
            return arr * lambda_ / (lambda_ + 2 * tau)
        else:
            return arr

    @pycu.redirect("arr", DASK=_prox_root)
    def _prox_sort(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        z = xp.sort(xp.abs(arr))[::-1]
        cumsum_z = xp.cumsum(z)
        test_array = z - (2 * tau / (1 + (xp.arange(z.size) + 1) * 2 * tau)) * cumsum_z
        if self.apply(arr) == 0:
            max_nzi = 0
        else:
            max_nzi = xp.max(xp.nonzero(test_array.reshape(-1) > 0)[0])
        threshold = cumsum_z[max_nzi]
        threshold *= 2 * tau / (1 + (max_nzi + 1) * 2 * tau)
        y = xp.fmax(0, xp.fabs(arr) - threshold)
        y *= xp.sign(arr)
        return y

    @pycu.vectorize("arr")
    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        arr = arr.ravel()
        if self._algo == "root":
            return self._prox_root(arr, tau)
        elif self._algo == "sort":
            return self._prox_sort(arr, tau)


class LInftyNorm(ShiftLossMixin, pyca.ProxFunc):
    r"""
    :math:`\ell_{\infty}`-norm, :math:`\Vert\mathbf{x}\Vert_2:=\max_{i=1,.,N} |x_i|`.
    """

    def __init__(self, dim: pyct.Integer = None):
        r"""
        Parameters
        ----------
        dim: pyct.Integer
            Dimension size. (Default: domain-agnostic.)
        """
        super().__init__(shape=(1, dim))
        self._lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.max(xp.fabs(arr), axis=-1, keepdims=True)
        return y

    @pycrt.enforce_precision(i=("arr", "tau"))
    @pycu.vectorize("arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        mu_max = self.apply(arr)
        if mu_max == 0:
            return arr
        else:
            func = lambda mu: xp.sum(xp.fmax(xp.fabs(arr) - mu, 0)) - tau
            mu_star = sopt.brentq(func, a=0, b=mu_max)
            y = xp.fmin(xp.fabs(arr), mu_star)
            y *= xp.sign(arr)
            return y
