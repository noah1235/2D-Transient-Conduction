from abc import ABC, abstractmethod
import numpy as np
from SRC.mat_props.tbl_linear_interp import interp_tabl


class Material_Prop(ABC):
    """
    Abstract base for temperature-dependent material properties.

    Implementations must be callable like:
        p_val = prop(T_i)

    where
        T_i : float or ndarray
            Temperature at a point or array of points.

    Returns
    -------
    float or ndarray
        Property value(s) at T_i with the same shape/broadcasting as input.
    """

    @abstractmethod
    def __call__(self, T_i):
        pass


class Const_Prop(Material_Prop):
    """
    Constant (temperature-independent) material property.

    Examples: density (ρ) or specific heat (c) when treated as constant.
    """

    def __init__(self, p):
        """
        Parameters
        ----------
        p : float
            Constant property value.
        """
        self.p = p

    def __call__(self, T_i):
        """
        Return the constant property with shape compatible to T_i.

        If T_i is an ndarray, return an array of identical values shaped like T_i.
        If T_i is scalar-like, return a scalar.
        """
        if isinstance(T_i, np.ndarray):
            return np.ones_like(T_i) * self.p
        else:
            return self.p


class Tbl_Prop(Material_Prop):
    """
    Table-based, linearly interpolated material property p(x).

    The table (x_tbl, y_tbl) is sorted at construction for safe interpolation.
    Extrapolation behavior is delegated to `interp_tabl`.
    """

    def __init__(self, x_tbl, y_tbl):
        """
        Parameters
        ----------
        x_tbl : array-like
            Independent variable samples (e.g., temperature).
        y_tbl : array-like
            Property values corresponding to x_tbl.
        """
        # Ensure monotonic x for interpolation
        sort_idx = np.argsort(x_tbl)
        self.y_tbl = y_tbl[sort_idx]
        self.x_tbl = x_tbl[sort_idx]

    def __call__(self, x):
        """
        Interpolate the property at x using the stored table.

        Parameters
        ----------
        x : float or ndarray
            Query point(s).

        Returns
        -------
        float or ndarray
            Interpolated property value(s). We return the first element of
            interp_tabl’s output to preserve the original behavior, assuming
            it returns (y, ...) such as (values, slopes) or similar.
        """
        return interp_tabl(self.x_tbl, self.y_tbl, x)[0]
