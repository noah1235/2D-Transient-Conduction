

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

def animate_trj(
    trj: np.ndarray,
    times: np.ndarray | None = None,
    dt: float | None = None,
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    fps: int = 20,
    stride: int = 1,
    extent: tuple[float, float, float, float] | None = None,  # (x0, x1, y0, y1)
    save: str | None = None,  # e.g. "movie.mp4" or "movie.gif"
    title: str = "Time",
):
    """
    Animate temporal evolution of a 3D array trj with shape (Nt, Nx, Ny).

    Args
    ----
    trj : np.ndarray
        Data array with shape (Nt, Nx, Ny).
    times : np.ndarray | None
        Optional time stamps of length Nt. If None, uses [0, 1, 2, ...]*dt or frame index.
    dt : float | None
        Time step (used if `times` is None). If both `times` and `dt` are None, uses frame index.
    cmap : str
        Colormap for imshow.
    vmin, vmax : float | None
        Color limits. If None, computed from finite values in `trj`.
    fps : int
        Frames per second for playback/saving.
    stride : int
        Use every `stride`-th frame to speed up.
    extent : tuple | None
        Imshow extent = (x0, x1, y0, y1). If None, pixel indices are used.
    save : str | None
        Filename to save animation (.mp4 or .gif). If None, just displays.
    title : str
        Base title text.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object (useful if you embed in notebooks).
    """
    assert trj.ndim == 3, "trj must have shape (Nt, Nx, Ny)"
    Nt, Nx, Ny = trj.shape

    # Frame selection
    frames_idx = np.arange(0, Nt, stride)

    # Handle time axis
    if times is not None:
        assert len(times) == Nt, "`times` length must match trj.shape[0]"
        t_vals = times[frames_idx]
        t_label = lambda k: f"Time = {t_vals[k]:.3g}"
    elif dt is not None:
        t_vals = frames_idx * dt
        t_label = lambda k: f"Time = {t_vals[k]:.3g}"
    else:
        t_label = lambda k: f"frame {frames_idx[k]}"

    # Color limits (ignore NaNs)
    if vmin is None or vmax is None:
        finite_vals = trj[np.isfinite(trj)]
        if finite_vals.size == 0:
            vmin = vmax = 0.0
        else:
            if vmin is None: vmin = np.percentile(finite_vals, 1)
            if vmax is None: vmax = np.percentile(finite_vals, 99)
            if vmin == vmax: vmax = vmin + 1e-9

    fig, ax = plt.subplots()
    im = ax.imshow(trj[frames_idx[0]].T, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=extent, animated=True,
                   )
    cbar = fig.colorbar(im, ax=ax, label="Temperature [k]")
    cbar.set_label("Temperature [k]", rotation=270, labelpad=15)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")  
    txt = ax.set_title(f"{t_label(0)}")

    def update(k):
        frame = frames_idx[k]
        im.set_data(trj[frame].T)
        txt.set_text(f"{t_label(k)}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frames_idx), interval=1000/fps, blit=True)

    if save is not None:
        if save.lower().endswith(".mp4"):
            # Requires ffmpeg installed
            anim.save(save, writer=FFMpegWriter(fps=fps, bitrate=2400))
        elif save.lower().endswith(".gif"):
            anim.save(save, writer=PillowWriter(fps=fps))
        else:
            raise ValueError("`save` must end with .mp4 or .gif")

    plt.show()
    return anim
