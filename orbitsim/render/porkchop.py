"""Render a porkchop delta-V grid to a PNG (offscreen) for use as a texture."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_porkchop_png(
    dv_total: np.ndarray,
    dep_times_s: np.ndarray,
    tof_grid_s: np.ndarray,
    path: str,
) -> str:
    """Write a filled-contour porkchop plot to ``path``; return ``path``.

    Parameters
    ----------
    dv_total : np.ndarray
        Total delta-V [m/s], shape (len(dep_times_s), len(tof_grid_s)); infeasible
        cells are ``np.inf`` and get masked out.
    dep_times_s : np.ndarray
        Departure times [s] (x-axis), plotted in days.
    tof_grid_s : np.ndarray
        Times of flight [s] (y-axis), plotted in days.
    path : str
        Output PNG path.

    Returns
    -------
    str
        ``path``.
    """
    masked = np.ma.masked_invalid(dv_total)
    x = dep_times_s / 86400.0
    y = tof_grid_s / 86400.0
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    cs = ax.contourf(x, y, (masked.T / 1000.0), levels=25)
    fig.colorbar(cs, ax=ax, label="total dv [km/s]")
    # Mark the grid minimum.
    if masked.count() > 0:
        flat = int(np.ma.argmin(masked))
        i, j = flat // masked.shape[1], flat % masked.shape[1]
        ax.plot(x[i], y[j], "r*", markersize=14, label="min dv")
        ax.legend(loc="upper right")
    ax.set_xlabel("departure [days]")
    ax.set_ylabel("time of flight [days]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
