import numpy as np

import pycsou.abc.operator as pyco
import pycsou.operator.blocks as pyblock
import pycsou.operator.linop.base as pybase
import pycsou.operator.linop.diff as pydiff
import pycsou.util.ptype as pyct
from pycsou.operator.linop import IdentityOp


class ManifoldGradient(pyco.LinOp):
    r"""
    Example
    -------

    .. plot::

        from pycsou.util.misc import peaks
        import numpy as np
        import matplotlib.pyplot as plt
        import copy
        import pycsou.operator.linop.diff as pydiff
        import pycsou.operator.diffusion as pydiffusion
        import pycsou.operator.linop.diff_manifold as pymanifold
        x=np.linspace(-2.5,2.5,50)
        y=np.linspace(-2.5,2.5,50)
        xv, yv = np.meshgrid(x, y)
        zv = peaks(xv, yv)*10
        x0=np.zeros(zv.shape)
        #x0[15,15]=1
        source_center = [0,-1.8]
        source_center2 = [-1,2]
        source_center3 = [0,0]
        source_center4 = [1,1.5]
        x0 = np.exp(-((xv-source_center[0])**2+(yv-source_center[1])**2)/0.1)+\
            np.exp(-((xv-source_center2[0])**2+(yv-source_center2[1])**2)/0.1)+\
            np.exp(-((xv-source_center3[0])**2+(yv-source_center3[1])**2)/0.1)+\
            np.exp(-((xv-source_center4[0])**2+(yv-source_center4[1])**2)/0.1)
        arg_shape=zv.shape
        ManifoldGradient = pymanifold.ManifoldGradient(arg_shape=arg_shape, manifold=zv)
        N_steps = 500
        gradient_param_space = pydiff.Gradient(arg_shape=zv.shape, diff_method="fd", sampling=1, mode="reflect", diff_type="forward")
        TikhonovDiffusivity = pydiffusion.TikhonovDiffusivity(arg_shape=arg_shape)
        TikhonovDiffusionCoeff = pydiffusion.DiffusionCoeffIsotropic(arg_shape=arg_shape, diffusivity=TikhonovDiffusivity)
        DivergenceDiffusionOp = pydiffusion.DivergenceDiffusionOp(arg_shape=arg_shape, gradient=gradient_param_space,
                                                                  diffusion_coefficient=TikhonovDiffusionCoeff)
        x_tikho = copy.deepcopy(x0.reshape(1, -1))
        x_manifold = copy.deepcopy(x0.reshape(1, -1))
        for n in range(N_steps):
            x_tikho -= 0.2*DivergenceDiffusionOp.grad(x_tikho)
            x_manifold -= 0.2*ManifoldGradient.gram_op(x_manifold)
        fig, ax = plt.subplots(2,2,figsize=(13, 8))
        p00=ax[0,0].imshow(x0)
        ax[0,0].set_title("Initial condition x0", fontsize=15)
        plt.colorbar(p00, ax=ax[0,0], fraction=0.04)

        p01=ax[0,1].imshow(zv)
        ax[0,1].contour(zv, colors='red', levels=30, linewidths=0.1)
        ax[0,1].set_title("Manifold M", fontsize=15)
        plt.colorbar(p01, ax=ax[0,1], fraction=0.04)

        p10=ax[1,0].imshow(x_tikho.reshape(x0.shape), interpolation="bilinear")
        ax[1,0].set_title("{}it Laplacian Diffusion".format(N_steps), fontsize=15)
        plt.colorbar(p10, ax=ax[1,0], fraction=0.04)

        p11=ax[1,1].imshow(x_manifold.reshape(x0.shape), interpolation="bilinear")
        ax[1,1].set_title("{}it Laplace-Beltrami Diffusion on M".format(N_steps), fontsize=15)
        plt.colorbar(p11, ax=ax[1,1], fraction=0.04)

    """

    def __init__(self, arg_shape: pyct.NDArrayShape, manifold: pyct.NDArray, **diff_kwargs):
        self.arg_shape = arg_shape
        self.size = int(np.prod(arg_shape))
        super().__init__(shape=((len(arg_shape) + 1) * self.size, self.size))
        # sanitize_kwargs() TODO
        # check manifold.shape==arg_shape
        self.grad_param_space = pydiff.Gradient(
            arg_shape=arg_shape, diff_method="fd", sampling=1, mode="reflect", diff_type="forward"
        )
        # instantiate G^{-1} op A
        # instantiate sqrt(det) DiagOp op B
        # instantiate 1/sqrt(det) DiagOp op C
        # instantiate J op D (or should we try J*G^{-1}?)
        # gradient_param_space op E
        # then
        # apply = D*E
        # adjoint = C*(E.T)*B*(D.T)
        # gram = C*(E.T)*B*A*E
        # compute gradient of third component of transformation w.r.t. parameter space
        mapping_gradient = self.grad_param_space.unravel(self.grad_param_space(manifold.reshape(1, -1))).squeeze()
        # compute determinant of first fundamental form
        det_G = 1 + np.linalg.norm(mapping_gradient, axis=0) ** 2
        # assemble DiagonalOps featuring determinant involved in adjoint expression
        vec = np.hstack((np.sqrt(det_G).reshape(-1), np.sqrt(det_G).reshape(-1)))
        self.det_sqrt_op = pybase.DiagonalOp(vec)
        self.inv_det_sqrt_op = pybase.DiagonalOp(1 / np.sqrt(det_G).reshape(-1))
        # assemble jacobian of the mapping
        identity_op = IdentityOp(len(arg_shape) * self.size)
        diag_op_J_1 = pybase.DiagonalOp(mapping_gradient[0, :, :].reshape(-1))
        diag_op_J_2 = pybase.DiagonalOp(mapping_gradient[1, :, :].reshape(-1))
        jacobian_third_block = pyblock.hstack([diag_op_J_1, diag_op_J_2])
        self.J_op = pyblock.vstack([identity_op, jacobian_third_block])
        # assemble G^{-1} operator
        diag_op_invG_1 = pybase.DiagonalOp((mapping_gradient[1, :, :].reshape(-1) ** 2 + 1) / det_G.reshape(-1))
        diag_op_invG_2 = pybase.DiagonalOp(
            (-mapping_gradient[0, :, :].reshape(-1) * mapping_gradient[1, :, :].reshape(-1)) / det_G.reshape(-1)
        )
        diag_op_invG_3 = diag_op_invG_2
        diag_op_invG_4 = pybase.DiagonalOp((mapping_gradient[0, :, :].reshape(-1) ** 2 + 1) / det_G.reshape(-1))
        op_row_1 = pyblock.hstack([diag_op_invG_1, diag_op_invG_2])
        op_row_2 = pyblock.hstack([diag_op_invG_3, diag_op_invG_4])
        self.invG_op = pyblock.vstack([op_row_1, op_row_2])
        # define apply_op, adjoint_op, gram_op
        self.apply_op = self.J_op * self.invG_op * self.grad_param_space
        self.adjoint_op = (
            self.inv_det_sqrt_op * (self.grad_param_space.T) * self.det_sqrt_op * self.invG_op * self.J_op.T
        )
        self.gram_op = (
            self.inv_det_sqrt_op * (self.grad_param_space.T) * self.det_sqrt_op * self.invG_op * self.grad_param_space
        )
        # # assemble G for arbitrary dimension
        # # Actually, we would more like need to assemble it as a set of matrices to solve the linear systems
        # for i in range(len(arg_shape)):
        #     ops_full_row = []
        #     ops_row_i = []
        #     for j in range(len(arg_shape)):
        #         vec = mapping_gradient[i, :, :].reshape(-1) * mapping_gradient[j, :, :].reshape(-1)
        #         if i == j:
        #             vec += 1
        #         ops_row_i.append(pybase.DiagonalOp(vec))
        #     ops_full_row.append(pyblock.hstack(ops_row_i))
        # self.G_op = pyblock.vstack(ops_full_row)

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply_op(arr)

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.adjoint_op(arr)

    def gram(self) -> pyct.NDArray:
        return self.gram_op
