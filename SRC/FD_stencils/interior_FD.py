from SRC.mat_props.thermal_cond import Thermal_Cond
from abc import ABC, abstractmethod


class Interior_FD(ABC):
    """
    Abstract base class for interior finite-difference (FD) operators that
    compute the divergence of heat flux (i.e., ∂/∂x (k ∂T/∂x) + ∂/∂y (k ∂T/∂y))
    or a compatible interior stencil contribution for 2D conduction.

    Implementations decide how to discretize spatial derivatives and how to
    handle temperature-dependent thermal conductivity k(T).
    """

    @abstractmethod
    def __init__(self, Nx, Ny, Delta_x, Delta_y, k: Thermal_Cond):
        """
        Parameters
        ----------
        Nx, Ny : int
            Grid resolution in x and y directions (number of interior nodes or cells).
        Delta_x, Delta_y : float
            Grid spacing in x and y directions.
        k : Thermal_Cond
            Thermal conductivity model. Must be callable on temperature and return
            (k(T), dk/dT(T)).
        """
        pass

    @abstractmethod
    def get_x_comp(self, i, j, T, Delta_x):
        """
        Compute the x-direction contribution at grid index (i, j).

        Parameters
        ----------
        i, j : int
            Grid indices.
        T : ndarray
            Temperature field array.
        Delta_x : float
            Grid spacing in x direction (passed explicitly here per interface).
        """
        pass

    @abstractmethod
    def get_y_comp(self, i, j, T, Delta_y):
        """
        Compute the y-direction contribution at grid index (i, j).

        Parameters
        ----------
        i, j : int
            Grid indices.
        T : ndarray
            Temperature field array.
        Delta_y : float
            Grid spacing in y direction (passed explicitly here per interface).
        """
        pass

    @abstractmethod
    def __call__(self, i, j, T, Delta_x, Delta_y):
        """
        Return the total interior contribution (x + y components) at (i, j).
        """
        pass


class Interior_Central_Diff_Stencil(Interior_FD):
    """
    Second-order central-difference interior stencil for 2D conduction with
    temperature-dependent thermal conductivity k(T).

    Discretization (per direction j ∈ {x, y}):
        ∂/∂xj ( k ∂T/∂xj ) ≈ (dk/dT|_i) * (∂T/∂xj)|_i  +  k_i * (∂²T/∂xj²)|_i

    where:
        (∂T/∂xj)|_i   ≈ (T_{i+1} - T_{i-1}) / (2 Δxj)             [CD_1D_2A]
        (∂²T/∂xj²)|_i ≈ (T_{i+1} + T_{i-1} - 2 T_i) / (Δxj²)      [CD_2D_2A]

    Boundary handling is expected to be performed externally; (i, j) should be
    valid interior indices with available neighbors.
    """

    def __init__(self, Nx, Ny, Delta_x, Delta_y, k: Thermal_Cond):
        # Store geometry and conductivity model
        self.k = k
        self.Nx = Nx
        self.Ny = Ny
        self.Delta_x = Delta_x
        self.Delta_y = Delta_y

    @staticmethod
    def CD_1D_2A(Delta_xj, T_ip1, T_im1):
        """
        Central-difference, second-order accurate, first derivative.

        (∂T/∂xj)|_i ≈ (T_{i+1} - T_{i-1}) / (2 Δxj)
        """
        return (T_ip1 - T_im1) / (2 * Delta_xj)

    @staticmethod
    def CD_2D_2A(Delta_xj, T_i, T_ip1, T_im1):
        """
        Central-difference, second-order accurate, second derivative.

        (∂²T/∂xj²)|_i ≈ (T_{i+1} + T_{i-1} - 2 T_i) / (Δxj²)
        """
        return (T_ip1 + T_im1 - 2 * T_i) / (Delta_xj**2)

    def stencil(self, Delta_xj, T_i, T_im1, T_ip1):
        """
        1D contribution for a single axis using central differences.

        Parameters
        ----------
        Delta_xj : float
            Grid spacing along the chosen axis.
        T_i : float
            Temperature at the current node.
        T_im1, T_ip1 : float
            Temperatures at immediate neighbors along the axis.

        Returns
        -------
        float
            Discrete approximation to ∂/∂xj ( k ∂T/∂xj ) at the node.
        """
        # k(T_i) model returns both conductivity and its temperature derivative
        k_i, dk_i__dT = self.k(T_i)

        # Product rule split into two terms:
        #   (dk/dT)_i * (∂T/∂xj)|_i  +  k_i * (∂²T/∂xj²)|_i
        return (
            dk_i__dT * self.CD_1D_2A(Delta_xj, T_ip1, T_im1)
            + k_i * self.CD_2D_2A(Delta_xj, T_i, T_ip1, T_im1)
        )

    def get_x_comp(self, i, j, T):
        """
        X-direction contribution at (i, j) using interior neighbors (i±1, j).

        Assumes (i-1, j) and (i+1, j) are valid interior indices.
        """
        return self.stencil(self.Delta_x, T[i, j], T[i - 1, j], T[i + 1, j])

    def get_y_comp(self, i, j, T):
        """
        Y-direction contribution at (i, j) using interior neighbors (i, j±1).

        Assumes (i, j-1) and (i, j+1) are valid interior indices.
        """
        return self.stencil(self.Delta_y, T[i, j], T[i, j - 1], T[i, j + 1])

    def __call__(self, i, j, T):
        """
        Total interior stencil contribution at (i, j):
            L[T]_ij = (x-component) + (y-component)
        """
        return self.get_x_comp(i, j, T) + self.get_y_comp(i, j, T)
