from abc import ABC, abstractmethod
import numpy as np
from SRC.mat_props.tbl_linear_interp import interp_tabl


class Thermal_Cond(ABC):
    """
    Abstract base for thermal conductivity models k(T).

    Implementations must be callable like:
        k_val, dk_dT = k_model(T_i)

    where
        T_i : float or ndarray
            Temperature(s) at which to evaluate the model.

    Returns
    -------
    (k, dk_dT) : tuple
        k     : float or ndarray, conductivity at T_i
        dk_dT : float or ndarray, derivative dk/dT at T_i
    """

    @abstractmethod
    def __call__(self, T_i):
        pass


class Const_k(Thermal_Cond):
    """
    Temperature-independent (constant) thermal conductivity.

    Returns (k, 0) regardless of input temperature.
    """

    def __init__(self, k):
        """
        Parameters
        ----------
        k : float
            Constant thermal conductivity value.
        """
        self.k = k

    def __call__(self, T_i):
        """
        Parameters
        ----------
        T_i : float or ndarray
            Temperature(s); unused for constant model.

        Returns
        -------
        (k, 0)
        """
        return self.k, 0


class TBL_K(Thermal_Cond):
    """
    Table-based, linearly interpolated thermal conductivity k(T).

    The (T_tbl, k_tbl) arrays are sorted at construction to ensure monotonic
    interpolation. The underlying `interp_tabl` is expected to return both
    the interpolated value and its derivative with respect to the independent
    variable (here, dk/dT), matching the interface used elsewhere:
        k, dk_dT = interp_tabl(T_tbl, k_tbl, T_i)
    """

    def __init__(self, k_tbl, T_tbl):
        """
        Parameters
        ----------
        k_tbl : array-like
            Sampled conductivity values corresponding to T_tbl.
        T_tbl : array-like
            Temperatures at which k_tbl is provided.
        """
        # Ensure temperature table is sorted ascending for interpolation
        sort_idx = np.argsort(T_tbl)
        self.k_tbl = k_tbl[sort_idx]
        self.T_tbl = T_tbl[sort_idx]

    def __call__(self, T_i):
        """
        Interpolate k and dk/dT at T_i using the stored tables.

        Parameters
        ----------
        T_i : float or ndarray
            Query temperature(s).

        Returns
        -------
        (k, dk_dT) : tuple of float or ndarray
            Interpolated conductivity and its temperature derivative.
        """
        return interp_tabl(self.T_tbl, self.k_tbl, T_i)
