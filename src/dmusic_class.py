import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.linalg import hankel
import sys, os
import h5py
from tkinter.filedialog import askopenfilename


class DMusic:
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
        self.t = t
        self.y = y
        self.set_params(poverlap, N, J, K, mesh_size, rlim, flim)

    def set_params(
        self, poverlap, N, J, K, mesh_size=(30, 500), rlim=(0, -1), flim=None
    ):
        self.poverlap = poverlap
        self.overlap = int(N * poverlap)
        self.N = N
        self.J = J
        self.K = K
        self.mesh_size = mesh_size
        self.rlim = rlim
        self.flim = flim
        self.delta_t = (self.t[-1] - self.t[0]) / (len(self.t) - 1)
        self.freq_nyq = round(1 / (2 * self.delta_t))

    def prepare_spgram(self):
        if self.flim is None:
            self.fs = np.linspace(0, self.freq_nyq, self.mesh_size[1])
        else:
            self.fs = np.linspace(
                self.flim[0], self.flim[1], self.mesh_size[1]
            )
        self.rs = np.linspace(self.rlim[0], self.rlim[1], self.mesh_size[0])
        self.k0 = np.arange(0, len(self.y) - self.N, self.N - self.overlap)
        self.ts = (
            np.array([self.t[i] for i in self.k0])
            + self.overlap * self.delta_t / 2
        )

    def make_rmat(self):
        ws = 2 * np.pi * self.fs
        rmat = np.zeros((self.J, len(self.rs) * len(ws)), dtype=np.complex128)
        for i_w in range(len(ws)):
            for i_r in range(len(self.rs)):
                r = np.exp(
                    np.arange(self.J)
                    * self.delta_t
                    * (self.rs[i_r] + 1j * ws[i_w])
                )
                norm_r = np.linalg.norm(r)
                rmat[:, len(self.rs) * i_w + i_r] = r / norm_r
        self.rmat = rmat
        return self.rmat

    def single_dmusic(self, k0i):
        ym = hankel(
            self.y[k0i : k0i + self.N - self.J],  # first column
            self.y[k0i + self.N - self.J - 1 : k0i + self.N - 1],  # last row
        )
        u, s, vh = np.linalg.svd(ym, compute_uv=True, full_matrices=True)
        vn = vh.conj().T[:, self.K :]
        tmp = (1 / np.linalg.norm((vn.T @ self.rmat), axis=0)).real
        tmp = tmp.reshape(self.mesh_size[1], self.mesh_size[0])
        tmp_i = np.abs(simpson(y=tmp, x=self.rs))
        return tmp_i

    def spgram_dmusic(self, mode="psd"):
        self.prepare_spgram()
        self.make_rmat()
        print("N_iter = {}\nPercentage completed:\n".format(len(self.k0)))
        ps = []
        last_percent = 0
        for i, k0i in enumerate(self.k0):
            percent = round(100 * i / len(self.k0))
            if percent != last_percent:
                print("{}%".format(percent), end=" ")
                sys.stdout.flush()
                last_percent = percent
            tmp_i = self.single_dmusic(k0i)
            ps.append(tmp_i)
        self.ps = np.array(ps).T
        if mode == "psd":
            self.ps = 10 * np.log10(np.abs(self.ps) / np.max(np.abs(self.ps)))
        return self.ps, self.fs, self.ts

    def plot_dmusic_sgram(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.pcolorfast(self.ts[[0, -1]], self.fs[[0, -1]], self.ps, cmap="jet")
        ax.set_xlabel("T [ms]")
        ax.set_ylabel("F (kHz)")
        return ax.get_figure(), ax

    def plot_in_context(self, ax=None, context=None):
        with plt.rc_context(context):
            fig, ax = self.plot_dmusic_sgram(ax)
        return fig, ax

    def save_dmusic(
        self,
        shot: int,
        savefile: str | None = None,
        savedir: str | None = None,
    ):
        """Save the DMUSIC result to an HDF5 file."""
        if savedir is None:
            savedir = "./dmusics/hdfs"
        if savefile is None:
            savefile = f"dmusic__{shot}__{self.t[0]:.2f}_{self.t[-1]:.2f}__{self.fs[0]:.2f}_{self.fs[-1]:.2f}.h5"
        filename = f"{savedir}/{savefile}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with h5py.File(filename, "w") as f:
            # Save original arrays
            f.create_dataset("t", data=self.t)
            f.create_dataset("y", data=self.y)
            # Save results
            f.create_dataset("ps", data=self.ps)
            f.create_dataset("fs", data=self.fs)
            f.create_dataset("ts", data=self.ts)
            # Save parameters
            params_group = f.create_group("params")
            params_group.create_dataset("N", data=self.N)
            params_group.create_dataset("J", data=self.J)
            params_group.create_dataset("K", data=self.K)
            params_group.create_dataset("poverlap", data=self.poverlap)
            params_group.create_dataset("rlim", data=self.rlim)
            params_group.create_dataset("flim", data=self.flim)
        return None

    @staticmethod
    def load_dmusic_class(file=None):
        """Load a DMUSIC object from an HDF5 file."""
        if file is None:
            file = askopenfilename(initialdir="./dmusics/hdfs")
        with h5py.File(file, "r") as f:
            # Load original arrays
            t = f["t"][:]  # type: ignore
            y = f["y"][:]  # type: ignore
            # Load results
            ps = f["ps"][:]  # type: ignore
            fs = f["fs"][:]  # type: ignore
            ts = f["ts"][:]  # type: ignore
            # Load parameters
            params_group = f["params"]
            N = params_group["N"][()]  # type: ignore
            J = params_group["J"][()]  # type: ignore
            K = params_group["K"][()]  # type: ignore
            poverlap = params_group["poverlap"][()]  # type: ignore
            rlim = params_group["rlim"][:]  # type: ignore
            flim = params_group["flim"][:]  # type: ignore
        #   Create DMUSIC object
        out = DMusic(
            t=t, y=y, N=N, J=J, K=K, poverlap=poverlap, rlim=rlim, flim=flim
        )
        out.ps = ps
        out.fs = fs
        out.ts = ts
        return out
