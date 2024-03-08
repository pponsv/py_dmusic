import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.linalg import hankel
import sys
from dmusic_class import DMusic


class DMusic_gpu(DMusic):
    def __init__(
        self,
        t,
        y,
        poverlap,
        N,
        J,
        K,
        mesh_size=(30, 500),
        rlim=(0, -1),
        flim=None,
    ):
        super().__init__(t, y, poverlap, N, J, K, mesh_size, rlim, flim)
        self.gpu_enabled = torch.cuda.is_available()
        self.set_dtype(torch.complex64)

    def set_dtype(self, dtype):
        self.dtype = dtype

    def make_rmat(self):
        rmat = super().make_rmat()
        if self.gpu_enabled:
            self.rmat = torch.tensor(rmat, dtype=torch.complex64).cuda()
        else:
            self.rmat = rmat
        return self.rmat

    def single_dmusic_gpus(self, new_ks):
        yms = []
        for k0i in new_ks:
            ym = hankel(
                self.y[k0i : k0i + self.N - self.J],  # first column
                self.y[
                    k0i + self.N - self.J - 1 : k0i + self.N - 1
                ],  # last row
            )
            yms.append(ym)
        yms = np.array(yms)
        yms = torch.tensor(yms, dtype=self.dtype).cuda()
        vn = torch.linalg.svd(yms, full_matrices=True)[2][
            :, self.K :, :
        ].conj()
        inv_norm_mat_product = (
            1
            / torch.linalg.norm(torch.linalg.matmul(vn, self.rmat), dim=1).real
        ).reshape(len(new_ks), self.mesh_size[1], self.mesh_size[0])
        batch_ps = np.array(
            [
                np.abs(simpson(y=tmpi.cpu().numpy(), x=self.rs, axis=1))
                for tmpi in inv_norm_mat_product
            ]
        )
        #   Memory management
        del vn, inv_norm_mat_product, yms
        torch.cuda.empty_cache()

        return batch_ps

    def spgram_dmusic(self, mode="psd"):
        if self.gpu_enabled is False:
            return super().spgram_dmusic(mode)

        self.prepare_spgram()
        self.make_rmat()

        print(f"N_iter = {len(self.k0)}\nPercentage completed:\n")
        mem_cutoff = torch.cuda.mem_get_info()[0]
        print("Free memory: ", torch.cuda.mem_get_info()[0] / 1e6, "MB")
        max_step_length = np.floor(
            mem_cutoff
            / (
                ((self.N - self.J) * self.mesh_size[0] * self.mesh_size[1])
                * 4
                * 2
            )
        ).astype(int)
        new_ks = np.array_split(
            self.k0, np.ceil(len(self.k0) / max_step_length)
        )

        print(len(self.k0), max_step_length)

        ps = []
        for idx_new_k, new_k in enumerate(new_ks):
            print(f"Processing batch {idx_new_k + 1}/{len(new_ks)}")
            tmp_ps = self.single_dmusic_gpus(new_k)
            ps.append(tmp_ps)

        self.ps = np.concatenate(ps, axis=0).T
        if mode == "psd":
            self.ps = 10 * np.log10(np.abs(self.ps) / np.max(np.abs(self.ps)))

        return self.ps, self.fs, self.ts
