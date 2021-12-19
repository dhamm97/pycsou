import itertools
import math
import numbers as nb
import typing as typ

import pycsou.abc as pyca
import pycsou.abc.operator as pyco
import pycsou.abc.solver as pycs
import pycsou.linop.base as pyclo
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct


class APGD(pycs.Solver):
    r"""
    Accelerated Proximal Gradient Descent (APGD) solver.

    APGD solves minimization problems of the form

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

    **Remark 2:** The convergence is guaranteed for step sizes :math:`\tau\leq 1/\beta`. Various
    acceleration schemes are described in [APGD]_. APGD achieves the following (optimal)
    *convergence rate* with the implemented acceleration scheme from Chambolle & Dossal:

    .. math::

       \lim\limits_{n\rightarrow \infty} n^2\left\vert \mathcal{J}(\mathbf{x}^\star)- \mathcal{J}(\mathbf{x}_n)\right\vert=0
       \qquad\&\qquad
       \lim\limits_{n\rightarrow \infty} n^2\Vert \mathbf{x}_n-\mathbf{x}_{n-1}\Vert^2_\mathcal{X}=0,

    for *some minimiser* :math:`{\mathbf{x}^\star}\in\arg\min_{\mathbf{x}\in\mathbb{R}^N} \;\left\{\mathcal{J}(\mathbf{x}):=\mathcal{F}(\mathbf{x})+\mathcal{G}(\mathbf{x})\right\}`.
    In other words, both the objective functional and the APGD iterates :math:`\{\mathbf{x}_n\}_{n\in\mathbb{N}}`
    converge at a rate :math:`o(1/n^2)`. Significant practical *speedup* can be achieved for values
    of :math:`d` in the range:math:`[50,100]` [APGD]_.
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
        log_var: pyct.VarName = ("primal",),
    ):
        super().__init__(
            folder=folder,
            exist_ok=exist_ok,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            log_var=log_var,
        )

        self._f = pyclo.NullOperator() if (f is None) else f
        self._g = pyclo.NullOperator() if (g is None) else g
        if (f is None) and (g is None):
            msg = " ".join(
                [
                    "Cannot minimize always-0 functional.",
                    "At least one of Parameter[f, g] must be specified.",
                ]
            )
            raise ValueError(msg)

    def fit(
        self,
        primal_init: pyct.NDArray,
        stop_crit: pycs.StoppingCriterion,
        mode: pycs.Mode = pycs.Mode.BLOCK,
        tau: typ.Optional[pyct.Number] = None,
        d: typ.Optional[pyct.Number] = 75,
    ):
        r"""
        Solve the minimization problem defined in :py:method:`APGD.__init__`, with the provided
        run-specifc parameters.

        Parameters
        ----------
        primal_init: NDArray
            (..., N) primal variable initial point(s).
        stop_crit: StoppingCriterion
            Stopping criterion to end solver iterations.
        mode: Mode
            Execution mode. See :py:class:`Solver` for usage examples.
            Useful method pairs depending on the execution mode:
            * BLOCK: fit()
            * ASYNC: fit() + busy() + stop()
            * MANUAL: fit() + steps()
        tau: Number
            Gradient step size. Defaults to :math:`1 / \beta` if unspecified.
        d: Number
            Chambolle & Dossal acceleration parameter :math:`d`. Should be greater than 2.
        """
        self.m_init(primal_init=primal_init, tau=tau, d=d)
        self._fit_init(mode, stop_crit)
        self._fit_run()

    def m_init(self, primal_init: pyct.NDArray, tau: pyct.Number, d: pyct.Number):
        self._mstate["primal"] = self._mstate["primal_prev"] = pycrt.coerce(primal_init)

        if tau is None:
            if math.isfinite(dl := self._f._diff_lipschitz):
                self._mstate["tau"] = pycrt.coerce(1 / dl)
            else:
                msg = "tau: automatic inference not supported for operators with unbounded Lipschitz gradients."
                raise ValueError(msg)
        else:
            try:
                assert tau > 0
                self._mstate["tau"] = pycrt.coerce(tau)
            except:
                raise ValueError(f"tau must be positive, got {tau}.")

        if d is None:
            self._mstate["a"] = itertools.repeat(pycrt.coerce(0))
        else:
            try:
                assert d > 2
                self._mstate["a"] = (pycrt.coerce(k / (k + 1 + d)) for k in itertools.count(start=0))
            except:
                raise ValueError(f"Expected d > 2, got {d}.")

    def m_step():
        mst = self._mstate  # shorthand

        a = next(mst["a"])
        y = (1 + a) * mst["primal"] - a * mst["primal_prev"]
        z = y - mst["tau"] * self._f.grad(y)

        mst["primal_prev"], mst["primal"] = mst["primal"], self._g.prox(z)

    def solution(self) -> pyct.NDArray:
        """
        Returns
        -------
        p: NDArray
            (..., N) primal solution.
        """
        _, data = self.stats()
        return data.get("primal")