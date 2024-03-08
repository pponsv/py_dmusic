import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import sys


def make_rmat(ws, rs, J, DT):
    rmat = np.zeros((J, len(rs) * len(ws)), dtype=np.complex128)
    nw = len(ws)
    nr = len(rs)
    for i_w in range(nw):
        for i_r in range(nr):
            r = np.exp(np.arange(J) * DT * (rs[i_r] + 1j * ws[i_w]))
            norm_r = np.linalg.norm(r)
            rmat[:, nr * i_w + i_r] = r / norm_r
    return rmat


def spgram_dmusic(
    t, y, poverlap, N, J, K, mesh_size=(30, 500), rlim=(0, -1), flim=None
):
    nr, nf = mesh_size
    overlap = int(N * poverlap)
    DT = (t[-1] - t[0]) / (len(t) - 1)
    fnyq = round(1 / (2 * DT))
    print("dt = {}, fnyq = {}".format(DT, fnyq))
    if flim is None:
        fs = np.linspace(0, fnyq, nf)
    else:
        fs = np.linspace(flim[0], flim[1], nf)
    ws = fs * 2 * np.pi
    rs = np.linspace(rlim[0], rlim[1], nr)
    k0 = np.arange(0, len(y) - N, N - overlap)
    ts = np.array([t[i] for i in k0]) + overlap * DT / 2
    rmat = make_rmat(ws, rs, J, DT)
    print("N_iter = {}\nPercentage completed:\n".format(len(k0)))
    ps = []
    last_percent = 0
    for i, k0i in enumerate(k0):
        percent = round(100 * i / len(k0))
        if percent != last_percent:
            print("{}%".format(percent), end=" ")
            sys.stdout.flush()
            last_percent = percent
        tmp_i = single_dmusic(y, N, J, K, mesh_size, rs, rmat, k0i)
        ps.append(tmp_i)
    ps = np.array(ps).T
    ps = 10 * np.log10(np.abs(ps) / np.max(np.abs(ps)))
    return ps, fs, ts


def single_dmusic(y, N, J, K, mesh_size, rs, rmat, k0i):
    y_tmp = y[k0i : k0i + N]
    ym = np.zeros((N - J, J))
    for i in range(N - J):
        ym[i] = y_tmp[i : J + i]
    u, s, v = np.linalg.svd(ym, compute_uv=True, full_matrices=True)
    vn = v.conj().T[:, K:]
    tmp = (1 / np.linalg.norm((vn.T @ rmat), axis=0)).real
    tmp = tmp.reshape(mesh_size[1], mesh_size[0])
    tmp_i = np.abs(trapezoid(y=tmp, x=rs))
    return tmp_i


def plot_dmusic_sgram(ps, fs, ts, figax=None, cmap="jet", **kwargs):
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(1, 1)
    ax.pcolorfast(ts, fs, ps, cmap=cmap, rasterized=True, **kwargs)
    return fig, ax


if __name__ == "__main__":
    x = np.linspace(0, 3, 3001)
    y = np.sin(2 * np.pi * 137 * x) + np.cos(2 * np.pi * 400 * x)

    res = spgram_dmusic(x, y, 0.9, 600, 300, 5)

    plot_dmusic_sgram(*res, cmap="viridis")

    plt.show()
