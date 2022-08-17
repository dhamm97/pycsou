import dask.array as da
import dask.distributed as dad
import numpy as np

import pycsou.operator.linop.nufft as nufft
import pycsou.runtime as pycrt
import pycsou.util as pycu

if __name__ == "__main__":
    use_dask = False
    nufft_kwargs = dict(
        real=False,
        eps=0,
        isign=-1,
        n_trans=5,
        nthreads=0,
    )

    rng = np.random.default_rng(0)
    D, M, N = 3, 200, 50
    x = rng.normal(size=(M, D)) + 2000
    z = rng.normal(size=(N, D))
    if use_dask:
        client = dad.Client(processes=False)  # processes=True yields a serialization error.
        x = da.from_array(x)
        z = da.from_array(z)

    xp = pycu.get_array_module(x)
    with pycrt.Precision(pycrt.Width.DOUBLE):
        A = nufft.NUFFT.type3(x, z, **nufft_kwargs)
        cB = A.complex_matrix(xp)
        rB = A.asarray(xp)

        arr = rng.normal(size=(34, nufft_kwargs["n_trans"], M))
        if not nufft_kwargs["real"]:
            arr = arr + 1j * rng.normal(size=arr.shape)
        if use_dask:
            arr = da.from_array(arr)

        rA_out_fw = A.apply(pycu.view_as_real(arr))
        cA_out_fw = pycu.view_as_complex(rA_out_fw)
        rB_out_fw = xp.tensordot(pycu.view_as_real(arr), rB, axes=[[2], [1]])
        cB_out_fw = xp.tensordot(arr, cB, axes=[[2], [1]])

        rA_out_bw = A.adjoint(rA_out_fw)
        if not nufft_kwargs["real"]:
            cA_out_bw = pycu.view_as_complex(rA_out_bw)
        rB_out_bw = xp.tensordot(rB_out_fw, rB.T, axes=[[2], [1]])
        cB_out_bw = xp.tensordot(cB_out_fw, cB.conj().T, axes=[[2], [1]])
        if nufft_kwargs["real"]:
            cB_out_bw = cB_out_bw.real

        res_fw_r = (xp.linalg.norm(rA_out_fw - rB_out_fw, axis=-1) / xp.linalg.norm(rB_out_fw, axis=-1)).max()
        res_fw_c = (xp.linalg.norm(cA_out_fw - cB_out_fw, axis=-1) / xp.linalg.norm(cB_out_fw, axis=-1)).max()
        res_bw_r = (xp.linalg.norm(rA_out_bw - rB_out_bw, axis=-1) / xp.linalg.norm(rB_out_bw, axis=-1)).max()
        if not nufft_kwargs["real"]:
            res_bw_c = (xp.linalg.norm(cA_out_bw - cB_out_bw, axis=-1) / xp.linalg.norm(cB_out_bw, axis=-1)).max()
        else:
            res_bw_c = res_bw_r

        res = dict(
            zip(
                ["res_fw_r", "res_fw_c", "res_bw_r", "res_bw_c", "eps"],
                pycu.compute(res_fw_r, res_fw_c, res_bw_r, res_bw_c, nufft_kwargs["eps"]),
            )
        )
        print(res)
