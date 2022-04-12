import itertools
import math
import typing as typ

import pycsou.abc.operator as pyco
import pycsou.abc.solver as pycs
import pycsou.operator.linop.base as pyclo
import pycsou.opt.stop as pycos
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct


class PGD(pycs.Solver):
    r"""
    Proximal Gradient Descent (PGD) solver.

    PGD solves minimization problems of the form

    .. math::

       {\min_{\mathbf{x}\in\mathbb{R}^N} \;\mathcal{F}(\mathbf{x})\;\;+\;\;\mathcal{G}(\mathbf{x})},

    where:

    * :math:`\mathcal{F}:\mathbb{R}^N\rightarrow \mathbb{R}` is *convex* and *differentiable*, with
      :math:`\beta`-*Lipschitz continuous* gradient, for some :math:`\beta\in[0,+\infty[`.
    * :math:`\mathcal{G}:\mathbb{R}^N\rightarrow \mathbb{R}\cup\{+\infty\}` is a *proper*, *lower
      semicontinuous* and *convex function* with a *simple proximal operator*.

    The problem is *feasible* -- i.e. there exists at least one solution.

    **Remark 1:** the algorithm is still valid if either :math:`\mathcal{F}` or :math:`\mathcal{G}`
    is zero.

    **Remark 2:** The convergence is guaranteed for step sizes :math:`\tau\leq 1/\beta`.

    **Remark 3:** Various acceleration schemes are described in [APGD]_. PGD achieves the following
    (optimal) *convergence rate* with the implemented acceleration scheme from Chambolle & Dossal:

    .. math::

       \lim\limits_{n\rightarrow \infty} n^2\left\vert \mathcal{J}(\mathbf{x}^\star)- \mathcal{J}(\mathbf{x}_n)\right\vert=0
       \qquad\&\qquad
       \lim\limits_{n\rightarrow \infty} n^2\Vert \mathbf{x}_n-\mathbf{x}_{n-1}\Vert^2_\mathcal{X}=0,

    for *some minimiser* :math:`{\mathbf{x}^\star}\in\arg\min_{\mathbf{x}\in\mathbb{R}^N} \;\left\{\mathcal{J}(\mathbf{x}):=\mathcal{F}(\mathbf{x})+\mathcal{G}(\mathbf{x})\right\}`.
    In other words, both the objective functional and the PGD iterates :math:`\{\mathbf{x}_n\}_{n\in\mathbb{N}}`
    converge at a rate :math:`o(1/n^2)`. Significant practical *speedup* can be achieved for values
    of :math:`d` in the range :math:`[50,100]` [APGD]_.

    **Remark 4:** The relative norm change of the primal variable is used as the default stopping criterion. By
    default, the algorithm stops when the norm of the difference between two consecutive PGD iterates
    :math:`\{\mathbf{x}_n\}_{n\in\mathbb{N}}` is smaller than 1e-4. Different stopping criteria can be used (see
    :py:mod:`~pycsou.opt.solver.stop`).


    **Parameterization** of ``PGD``

    f: DiffFunc | None
        Differentiable function, instance of :py:class:`~pycsou.abc.operator.DiffFunc`.
    g: ProxFunc | None
        Proximable function, instance of :py:class:`~pycsou.abc.operator.ProxFunc`.


    **Parameterization** of ``PGD.fit()``

    x0: NDArray
        (..., N) initial point(s).
    tau: Real
        Gradient step size.
        Defaults to :math:`1 / \beta` if unspecified.
    acceleration: bool
        If True (default), then use Chambolle & Dossal acceleration scheme.
    d: Real
        Chambolle & Dossal acceleration parameter :math:`d`.
        Should be greater than 2.
        Only meaningful if `acceleration` is True.
        Defaults to 75 in unspecified.
    """

    def __init__(
        self,
        f: typ.Optional[pyco.DiffFunc] = None,
        g: typ.Optional[pyco.ProxFunc] = None,
        *,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        show_progress: bool = True,
        log_var: pyct.VarName = ("x",),
    ):
        super().__init__(
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )

        self._f = pyclo.NullFunc() if (f is None) else f
        self._g = pyclo.NullFunc() if (g is None) else g
        if (f is None) and (g is None):
            msg = " ".join(
                [
                    "Cannot minimize always-0 functional.",
                    "At least one of Parameter[f, g] must be specified.",
                ]
            )
            raise ValueError(msg)

    @pycrt.enforce_precision(i=["x0", "tau"])
    def m_init(
        self,
        x0: pyct.NDArray,
        tau: typ.Optional[pyct.Real] = None,
        acceleration: bool = True,
        d: typ.Optional[pyct.Real] = 75,
    ):
        mst = self._mstate  # shorthand
        mst["x"] = mst["x_prev"] = x0

        if tau is None:
            if math.isfinite(dl := self._f._diff_lipschitz):
                mst["tau"] = pycrt.coerce(1 / dl)
            else:
                msg = "tau: automatic inference not supported for operators with unbounded Lipschitz gradients."
                raise ValueError(msg)
        else:
            try:
                assert tau > 0
                mst["tau"] = tau
            except:
                raise ValueError(f"tau must be positive, got {tau}.")

        if acceleration:
            try:
                assert d > 2
                mst["a"] = (pycrt.coerce(k / (k + 1 + d)) for k in itertools.count(start=0))
            except:
                raise ValueError(f"Expected d > 2, got {d}.")
        else:
            mst["a"] = itertools.repeat(pycrt.coerce(0))

    def m_step(self):
        mst = self._mstate  # shorthand

        a = next(mst["a"])
        y = (1 + a) * mst["x"] - a * mst["x_prev"]
        z = y - mst["tau"] * self._f.grad(y)

        mst["x_prev"], mst["x"] = mst["x"], self._g.prox(z, mst["tau"])

    def default_stop_crit(self) -> pycs.StoppingCriterion:
        stop_crit = pycos.RelError(
            eps=1e-4,
            var="x",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        p: NDArray
            (..., N) solution.
        """
        data, _ = self.stats()
        return data.get("x")
