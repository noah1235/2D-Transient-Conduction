import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib as mpl
from SRC.plotting.utils import save_svg, set_grid

def plot_q_residual_T_BC(
    BC_debug, t_eval,
    q_label="Boundary heat flux", q_units="W/m²",
    T_label="Boundary temperature", T_units="K",
    boundaries=None,  # e.g., ("left","right"); default = all
    save_path=None
):
    """
    Parameters
    ----------
    BC_debug : array-like, shape (N_times, 4, >=2)
        [:, side, 0] -> T (left=0, top=1, right=2, bottom=3)
        [:, side, 1] -> q
        [:, side, 2] -> conv (optional)
        If a selected side's conv has any NaNs, its residual is not plotted.
    t_eval : array-like, shape (N_times,)
        Time vector.
    boundaries : tuple[str] | list[str] | None
        Subset of {"left","top","right","bottom"} to plot, e.g. ("left","right").
        If None, plots all four.

    Returns
    -------
    fig, (ax_q, ax_res, ax_T)
    """
    BC_debug = np.asarray(BC_debug); t_eval = np.asarray(t_eval)

    if BC_debug.ndim != 3 or BC_debug.shape[1] != 4 or BC_debug.shape[2] < 2:
        raise ValueError(f"BC_debug must have shape (N, 4, >=2); got {BC_debug.shape}")
    if BC_debug.shape[0] != t_eval.shape[0]:
        raise ValueError("t_eval and BC_debug length mismatch")

    name_to_idx = {"left": 0, "top": 1, "right": 2, "bottom": 3}
    idx_to_pretty = {0: "Left", 1: "Top", 2: "Right", 3: "Bottom"}

    if boundaries is None:
        sel_names = ("left", "top", "right", "bottom")
    else:
        # Normalize and validate
        sel_names = tuple(str(b).lower() for b in boundaries)
        invalid = [b for b in sel_names if b not in name_to_idx]
        if invalid:
            raise ValueError(f"Invalid boundary names: {invalid}. "
                             f"Valid options: {list(name_to_idx.keys())}")

    # Extract series (vectorized access)
    T = BC_debug[:, :, 0]                # (N, 4)
    q = BC_debug[:, :, 1]                # (N, 4)
    conv = (BC_debug[:, :, 2]
            if BC_debug.shape[2] >= 3
            else np.full_like(q, np.nan, dtype=float))  # (N, 4)

    fig, (ax_q, ax_res, ax_T) = plt.subplots(
        1, 3, figsize=(16, 4.8), sharex=True, constrained_layout=True
    )


    # --- Panel 1: q(t)
    line_handles = {}
    for b in sel_names:
        j = name_to_idx[b]
        lh = ax_q.plot(t_eval, q[:, j], label=f"{idx_to_pretty[j]} (q)", zorder=2)[0]
        line_handles[j] = lh  # store for color reuse

    ax_q.set_xlabel("Time")
    ax_q.set_ylabel(f"{q_label} [{q_units}]")
    ax_q.set_title("Boundary Heat Flux q(t)")

    # → Enable scientific notation for y-axis
    ax_q.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_q.yaxis.get_offset_text().set_fontsize(9)  # optional, smaller offset text
    set_grid(ax_q)
    if sel_names:
        ax_q.legend(loc="best")

    # --- Panel 2: residual = conv - q (only if no NaNs for that side)
    markevery = max(1, len(t_eval)//25)
    stroke = [pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()]

    any_residual = False
    for b in sel_names:
        j = name_to_idx[b]
        conv_j = conv[:, j]
        if not np.isnan(conv_j).any():
            resid = conv_j - q[:, j]
            ax_res.plot(
                t_eval, resid,
                color=line_handles[j].get_color(),
                label=f"{idx_to_pretty[j]} (conv − q)",
                zorder=3, path_effects=stroke
            )
            any_residual = True

    ax_res.axhline(0.0, linewidth=1.0, linestyle=":", color="black", alpha=0.6)
    ax_res.set_xlabel("Time")
    ax_res.set_ylabel(f"Residual [{q_units}]")
    ax_res.set_title("Residual: (conv − q)")

    # → Enable scientific notation for residual plot as well
    ax_res.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_res.yaxis.get_offset_text().set_fontsize(9)
    set_grid(ax_res)
    if any_residual:
        ax_res.legend(loc="best")

    # --- Panel 3: T(t)
    for b in sel_names:
        j = name_to_idx[b]
        ax_T.plot(t_eval, T[:, j], label=idx_to_pretty[j],
                  color=line_handles[j].get_color())
    ax_T.set_xlabel("Time")
    ax_T.set_ylabel(f"{T_label} [{T_units}]")
    ax_T.set_title("Boundary Temperature T(t)")
    set_grid(ax_T)
    if sel_names:
        ax_T.legend(loc="best")

    if save_path.lower().endswith(".svg"):
        save_svg(mpl, fig, save_path)

    return fig, (ax_q, ax_res, ax_T)
