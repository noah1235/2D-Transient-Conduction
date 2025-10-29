import numpy as np

def interp_tabl(x_tbl, y_tbl, x, *, assume_sorted=False, extrapolate=False):
    """
    Piecewise-linear interpolation with derivative.

    Parameters
    ----------
    x_tbl : array_like
        Sample locations (independent variable).
    y_tbl : array_like
        Sample values (dependent variable), same length as x_tbl.
    x : float or array_like
        Query point(s).
    assume_sorted : bool, optional
        If False (default), (x_tbl, y_tbl) are sorted by x_tbl first.
    extrapolate : bool, optional
        If True, linearly extrapolate beyond the table ends.
        If False (default), clamp to nearest endpoint (derivative = slope of end segment).

    Returns
    -------
    y : float or np.ndarray
        Interpolated value(s) at x.
    dydx : float or np.ndarray
        Piecewise-constant derivative(s) dy/dx at x.
    """
    x_tbl = np.asarray(x_tbl, dtype=float)
    y_tbl = np.asarray(y_tbl, dtype=float)
    xq = np.asarray(x, dtype=float)

    if x_tbl.ndim != 1 or y_tbl.ndim != 1:
        raise ValueError("x_tbl and y_tbl must be 1-D.")
    if x_tbl.size != y_tbl.size:
        raise ValueError("x_tbl and y_tbl must have the same length.")
    if x_tbl.size == 0:
        raise ValueError("Empty lookup table.")
    if x_tbl.size == 1:
        # Only one point: value is constant, derivative zero
        y = np.full_like(xq, y_tbl[0], dtype=float)
        dydx = np.zeros_like(xq, dtype=float)
        return (y.item(), dydx.item()) if np.isscalar(x) else (y, dydx)

    # Sort by x if needed
    if not assume_sorted:
        order = np.argsort(x_tbl)
        x_tbl = x_tbl[order]
        y_tbl = y_tbl[order]

    # Find insertion indices: j so that x_tbl[j-1] <= x < x_tbl[j]
    j = np.searchsorted(x_tbl, xq, side="left")

    # Handle extrapolation/clamping
    if extrapolate:
        # Clamp the segment indices to [1, n-1] but allow x outside for alpha
        j = np.clip(j, 1, x_tbl.size - 1)
    else:
        # Clamp x to table range first, then recompute j
        xq = np.clip(xq, x_tbl[0], x_tbl[-1])
        j = np.searchsorted(x_tbl, xq, side="left")
        j = np.clip(j, 1, x_tbl.size - 1)

    x0 = x_tbl[j - 1]
    x1 = x_tbl[j]
    y0 = y_tbl[j - 1]
    y1 = y_tbl[j]

    # Slopes; guard zero-length segments
    dx = (x1 - x0)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = (y1 - y0) / dx
    # Where dx == 0 (duplicate x’s), treat as constant segment
    mask_zero = (dx == 0)
    if np.any(mask_zero):
        slope = np.where(mask_zero, 0.0, slope)

    # Linear interpolation
    alpha = np.zeros_like(xq, dtype=float)
    nonzero = ~mask_zero
    alpha[nonzero] = (xq[nonzero] - x0[nonzero]) / dx[nonzero]
    y = (1.0 - alpha) * y0 + alpha * y1
    dydx = slope

    # Preserve scalar if input was scalar
    if np.isscalar(x):
        return float(y.item()), float(dydx.item())
    return y, dydx
