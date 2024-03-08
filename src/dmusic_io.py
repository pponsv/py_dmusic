import pickle
import os
from . import dmusic_functions, dmusic_class
import plotting_styles as ps
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename, askopenfilenames
import h5py


def show_dmusic(file=None, savefile=None):
    if file is None:
        file = askopenfilename(initialdir="./dmusics/pkls")
    out = load_pickle(file)
    shot = int(os.path.basename(file).split("_")[1])
    print([i.shape for i in out])

    with ps.rc_context(ps.pub_style_one_cantarell):
        fig, ax = plt.subplots(1, 1, figsize=[6, 3])
        fig, ax = dmusic_functions.plot_dmusic_sgram(*out, figax=(fig, ax))
        ax.set(xlabel="T [ms]", ylabel="Freq [kHz]", title=f"#{shot}")
    # plt.show()
    if savefile is not None:
        fig.savefig(savefile, dpi=600)
    return fig, ax, shot, *out


def save_dmusic(out, shot, tlim, flim) -> None:
    ftlim = f"{tlim[0]:.2f}_{tlim[1]:.2f}"
    fflim = f"{flim[0]:.2f}_{flim[1]:.2f}"
    path = f"./dmusics/pkls/dmusic_{shot}__{ftlim}__{fflim}.pkl"
    with open(path, "wb") as pfile:
        pickle.dump(out, pfile)


def load_dmusic(shot, tlim, flim):
    ftlim = f"{tlim[0]:.2f}_{tlim[1]:.2f}"
    fflim = f"{flim[0]:.2f}_{flim[1]:.2f}"
    path = f"./dmusics/dmusic_{shot}__{ftlim}__{fflim}.pkl"
    return load_pickle(path)


def load_pickle(path):
    with open(path, "rb") as pfile:
        out = pickle.load(pfile)
    return out
