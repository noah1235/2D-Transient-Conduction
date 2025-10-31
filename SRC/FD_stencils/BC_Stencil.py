from abc import ABC, abstractmethod
from SRC.BCs.Temp_BC import Temp_BC
from SRC.FD_stencils.interior_FD import Interior_FD
from scipy.optimize import newton
import numpy as np


class BC_Stencil(ABC):
    """
    Abstract base for boundary-condition (BC) stencils.

    Responsibilities:
    - Route to the correct (upwind/backwind) BC controller based on boundary side.
    - Provide a unified interface to compute the boundary contribution and
      (optionally) populate debug arrays with boundary temperature and heat flux.
    - Delegate the actual 1D boundary stencil math to concrete subclasses
      (e.g., temperature BC, heat-flux BC, convection BC).
    """

    def set_side(self, side):
        """
        Set which boundary this stencil instance is attached to.

        Parameters
        ----------
        side : {"left", "right", "top", "bot"}
            The domain side; determines upwind/backwind orientation and axis.
        """
        self.side = side

    @abstractmethod
    def upwind_BC_ctrl(self, T, i, j, t):
        """
        Handle BC computation for right/top boundaries (upwind direction).

        Returns
        -------
        comp : float
            Discrete approximation for ∂/∂xj ( k ∂T/∂xj ) at the interior node.
        T_boundary : float
            Reconstructed temperature at the physical boundary.
        boundary_heat_flux : float
            Heat flux at the boundary (Fourier's law) with outward normal sign.
        """
        pass

    @abstractmethod
    def backwind_BC_ctrl(self, T, i, j, t):
        """
        Handle BC computation for left/bottom boundaries (backwind direction).

        Returns
        -------
        (comp, T_boundary, boundary_heat_flux) as in upwind_BC_ctrl.
        """
        pass

    @abstractmethod
    def backwind_BC_stencil(Delta_xj, T_i, T_ip1, T_ip2, BC_dict):
        """
        1D boundary stencil (backwind) along axis j.

        Parameters
        ----------
        Delta_xj : float
            Grid spacing along the boundary-normal axis.
        T_i, T_ip1, T_ip2 : float
            Center and forward neighbor temperatures along axis j.
        BC_dict : dict
            Boundary data (depends on subclass), e.g., {"T_BC": ...} or {"q": ...}.

        Returns
        -------
        dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj
        """
        pass

    @abstractmethod
    def upwind_BC_stencil(Delta_xj, T_i, T_im1, T_im2, BC_dict):
        """
        1D boundary stencil (upwind) along axis j.

        Parameters mirror backwind_BC_stencil but use backward neighbors.
        """
        pass

    @abstractmethod
    def get_BC(self, i, t):
        """
        Retrieve boundary condition data at index 'i' (span-wise) and time 't'.

        Returns
        -------
        dict
            Keys depend on subclass, e.g., {"T_BC": ...}, {"q": ...}, {"h": ..., "T_inf": ...}
        """
        pass

    def get_BC_comp(self, T, i, j, t, T_boundary_list=None, q_boundary_list=None, conv_boundary_list=None):
        """
        Compute the BC contribution at interior node (i, j) and optionally log diagnostics.

        The side determines whether we call backwind (left/bot) or upwind (right/top)
        stencil controllers. If diagnostic arrays are provided, we write:
          - T_boundary_list: reconstructed T at boundary
          - q_boundary_list: boundary heat flux (positive sign convention handled below)
          - conv_boundary_list: for convection BCs only, h*(T_inf - T_boundary)

        Notes
        -----
        Heat-flux sign 'q_sign':
          - For right/top (outward normal pointing +x or +y), we flip sign (q_sign = -1)
            to keep a consistent outward-positive convention in q_boundary_list.
        """
        if self.side == "left" or self.side == "bot":
            comp, T_boundary, boundary_heat_flux = self.backwind_BC_ctrl(T, i, j, t)
        elif self.side == "right" or self.side == "top":
            comp, T_boundary, boundary_heat_flux = self.upwind_BC_ctrl(T, i, j, t)

        BC_debug = (T_boundary_list is not None) and (q_boundary_list is not None)
        if BC_debug:
            # Sign convention for outward heat flux on different sides
            q_sign = 1
            if self.side == "right" or self.side == "top":
                q_sign = -1

            # Choose index along span-wise direction: j for vertical edges, i for horizontal edges
            if self.side == "left" or self.side == "right":
                T_boundary_list[j] = T_boundary
                q_boundary_list[j] = q_sign * boundary_heat_flux

                # For convection, also record h*(T_inf - T_b) at this boundary location
                if isinstance(self, Conv_BC_Stencil):
                    h, T_inf = self.conv_BC[j, t]
                    conv_boundary_list[j] = h * (T_inf - T_boundary)
            else:
                T_boundary_list[i] = T_boundary
                q_boundary_list[i] = q_sign * boundary_heat_flux
                if isinstance(self, Conv_BC_Stencil):
                    h, T_inf = self.conv_BC[i, t]
                    conv_boundary_list[i] = h * (T_inf - T_boundary)

        return comp

    def get_int_comp(self, T, i, j):
        """
        Interior contribution adjacent to the boundary at (i, j), projected on
        the boundary-normal axis. We reuse the interior stencil but select the
        component perpendicular to the boundary:

        - left/right boundaries → use y-component (normal is ±x; interior term uses the other axis)
        - top/bottom boundaries → use x-component
        """
        if self.side == "left":
            return self.int_stencil.get_y_comp(i, j, T)
        elif self.side == "bot":
            return self.int_stencil.get_x_comp(i, j, T)
        elif self.side == "right":
            return self.int_stencil.get_y_comp(i, j, T)
        elif self.side == "top":
            return self.int_stencil.get_x_comp(i, j, T)

    def upwind_BC_ctrl(self, T, i, j, t):
        """
        Controller for right/top boundaries (use backward neighbors along axis j).

        Builds local stencil inputs, calls the upwind BC stencil, and returns:
        (interior operator contribution, reconstructed boundary temperature, boundary heat flux).
        """
        if self.side == "left" or self.side == "bot":
            raise ValueError("Wrong orientation")

        T_i = T[i, j]
        if self.side == "right":
            BC_dict = self.get_BC(j, t)
            T_im1 = T[i-1, j]
            T_im2 = T[i-2, j]
            Delta_xj = self.Delta_x

        elif self.side == "top":
            BC_dict = self.get_BC(i, t)
            T_im1 = T[i, j-1]
            T_im2 = T[i, j-2]
            Delta_xj = self.Delta_y

        # k and dk/dT at interior point
        k_i, dk_i__dT = self.k(T_i)

        # 1D upwind boundary stencil returns derivative approximations at interior and at boundary
        dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj = self.upwind_BC_stencil(
            Delta_xj, T_i, T_im1, T_im2, BC_dict
        )

        # Fourier's law at the boundary (evaluate k at T_boundary)
        k_boundary = self.k(T_boundary)[0]
        boundary_heat_flux = -k_boundary * dT_b__dxj

        # Product rule split: (dk/dT)*dT/dx + k*d2T/dx2
        return dk_i__dT * dT_i__dxj + k_i * d2T_i__dxj2, T_boundary, boundary_heat_flux

    def backwind_BC_ctrl(self, T, i, j, t):
        """
        Controller for left/bottom boundaries (use forward neighbors along axis j).
        """
        if self.side == "right" or self.side == "top":
            raise ValueError("Wrong orientation")

        T_i = T[i, j]
        if self.side == "left":
            BC_dict = self.get_BC(j, t)
            T_ip1 = T[i+1, j]
            T_ip2 = T[i+2, j]
            Delta_xj = self.Delta_x

        elif self.side == "bot":
            BC_dict = self.get_BC(i, t)
            T_ip1 = T[i, j+1]
            T_ip2 = T[i, j+2]
            Delta_xj = self.Delta_y

        k_i, dk_i__dT = self.k(T_i)

        dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj = self.backwind_BC_stencil(
            Delta_xj, T_i, T_ip1, T_ip2, BC_dict
        )

        k_boundary = self.k(T_boundary)[0]
        boundary_heat_flux = -k_boundary * dT_b__dxj

        return dk_i__dT * dT_i__dxj + k_i * d2T_i__dxj2, T_boundary, boundary_heat_flux


class Temp_BC_Stencil(BC_Stencil):
    """
    Dirichlet/temperature boundary condition stencil.

    Enforces T = T_BC at the physical boundary via a high-order reconstruction
    that also yields approximations for dT/dxj and d²T/dxj² at the interior node.
    """

    def __init__(self, Delta_x, Delta_y, int_stencil: Interior_FD, T_BC: Temp_BC, k):
        self.Delta_x = Delta_x
        self.Delta_y = Delta_y
        self.T_BC = T_BC          # callable/indexable temperature BC
        self.k = k                # thermal conductivity model k(T) → (k, dk/dT)
        self.int_stencil = int_stencil

        # Optional debug arrays (shape compatible with span-wise index)
        self.T_boundary_debug = np.zeros_like(T_BC)
        self.q_boundary_debug = np.zeros_like(T_BC)

    def get_BC(self, idx, t):
        """Return the boundary temperature at span-wise index and time."""
        return {"T_BC": self.T_BC[idx, t]}

    @staticmethod
    def upwind_BC_stencil(Delta_xj, T_i, T_im1, T_im2, BC_dict):
        """
        Upwind (right/top) Dirichlet stencil. Reconstructs boundary and derivatives
        using interior point and two backward neighbors.
        """
        T_bc = BC_dict["T_BC"]

        # Shifted variables around T_i (a = T_b - T_i, b/c = neighbor deltas)
        a = T_bc - T_i
        b = T_im1 - T_i
        c = T_im2 - T_i

        # Polynomial coefficients for 3rd-order reconstruction (pre-derived)
        y = (16/15)*a - (2/3)*b + (1/10)*c
        z = (16/5)*a  + 2*b     - (1/5)*c
        w = (16/5)*a  + 4*b     - (6/5)*c

        # Interior derivatives at i
        dT_i__dxj   = y / Delta_xj
        d2T_i__dxj2 = z / (Delta_xj**2)
        d3T_i__dxj3 = w / (Delta_xj**3)

        # Boundary reconstruction and boundary derivative
        T_boundary = T_i + (1/2)*y + (1/8)*z + (1/48)*w
        dT_b__dxj  = dT_i__dxj + (1/2)*d2T_i__dxj2*Delta_xj + (1/8)*d3T_i__dxj3*(Delta_xj**2)

        return dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj

    @staticmethod
    def backwind_BC_stencil(Delta_xj, T_i, T_ip1, T_ip2, BC_dict):
        """
        Backwind (left/bottom) Dirichlet stencil. Uses interior point and two forward neighbors.
        """
        T_bc = BC_dict["T_BC"]

        a = T_bc - T_i
        b = T_ip1 - T_i
        c = T_ip2 - T_i

        y = -(16/15)*a + (2/3)*b - (1/10)*c
        z =  (16/5)*a  + 2*b     - (1/5)*c
        w = -(16/5)*a  - 4*b     + (6/5)*c

        dT_i__dxj   = y / Delta_xj
        d2T_i__dxj2 = z / (Delta_xj**2)
        d3T_i__dxj3 = w / (Delta_xj**3)

        # Boundary recon (note signs vs. upwind)
        T_boundary = T_i - (1/2)*y + (1/8)*z - (1/48)*w
        dT_b__dxj  = dT_i__dxj - (1/2)*d2T_i__dxj2*Delta_xj + (1/8)*d3T_i__dxj3*(Delta_xj**2)

        return dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj


class Heat_Flux_BC_Stencil(BC_Stencil):
    """
    Neumann/heat-flux boundary condition stencil.

    Enforces -k(T_b) * (∂T/∂n)_b = q by solving for an auxiliary parameter 'a'
    that enters the boundary reconstruction; 'a' is solved with a scalar Newton
    iteration per boundary location (nonlinear due to k(T_b)).
    """

    def __init__(self, Delta_x, Delta_y, int_stencil: Interior_FD, q_BC, k):
        self.Delta_x = Delta_x
        self.Delta_y = Delta_y
        self.q_BC = q_BC          # callable/indexable heat-flux BC q(x, t)
        self.k = k                # k(T) model
        self.int_stencil = int_stencil

    def get_BC(self, idx, t):
        """Return boundary heat flux at span-wise index and time."""
        return {"q": self.q_BC[idx, t]}

    def a_fn(self, T_b, Delta_xj, BC_dict):
        """
        Map boundary temperature → auxiliary parameter a(T_b).

        Derivation:
          For these polynomials, 'a' is proportional to Δx * q / k(T_b)
          (up to a sign handled in upwind/backwind wrappers).
        """
        q = BC_dict["q"]
        k_b = self.k(T_b)[0]
        return -((Delta_xj * q) / k_b)

    def backwind_BC_stencil(self, Delta_xj, T_i, T_ip1, T_ip2, BC_dict):
        """
        Backwind (left/bottom) Neumann stencil.

        Solves a = a(T_b) using Newton, where T_b depends on (y, z, w),
        which in turn depend linearly on a; then reconstructs derivatives.
        """
        b = T_ip1 - T_i
        c = T_ip2 - T_i
        T_b_fn = lambda y, z, w: T_i - (1/2)*y + (1/8)*z - (1/48)*w

        # Linear maps from 'a' to polynomial coefficients
        y_fn = lambda a: (16*a + 44*b - 7*c)/46
        z_fn = lambda a: (-24*a + 26*b - c)/23
        w_fn = lambda a: (24/23) * (a - 3*b + c)

        def F(a):
            # Root function enforcing a = a_fn(T_b(a))
            y = y_fn(a); z = z_fn(a); w = w_fn(a)
            T_b = T_b_fn(y, z, w)
            return a - self.a_fn(T_b, Delta_xj, BC_dict)

        # Initial guess from interior temperature
        a0 = self.a_fn(T_i, Delta_xj, BC_dict)
        a = newton(F, x0=a0)

        # Recompute coefficients at converged 'a'
        y = y_fn(a); z = z_fn(a); w = w_fn(a)

        # Interior derivatives
        dT_i__dxj   = y / Delta_xj
        d2T_i__dxj2 = z / (Delta_xj**2)
        d3T_i__dxj3 = w / (Delta_xj**3)

        # Boundary reconstruction and derivative
        T_boundary = T_b_fn(y, z, w)
        dT_b__dxj  = dT_i__dxj - (1/2)*d2T_i__dxj2*Delta_xj + (1/8)*d3T_i__dxj3*(Delta_xj**2)

        return dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj

    def upwind_BC_stencil(self, Delta_xj, T_i, T_im1, T_im2, BC_dict):
        """
        Upwind (right/top) Neumann stencil.

        Same idea as backwind, with appropriate neighbor orientation/signs.
        """
        b = T_im1 - T_i
        c = T_im2 - T_i
        T_b_fn = lambda y, z, w: T_i + (1/2)*y + (1/8)*z + (1/48)*w

        y_fn = lambda a: (16*a - 44*b + 7*c)/46
        z_fn = lambda a: (24*a + 26*b - c)/23
        w_fn = lambda a: (24/23) * (a + 3*b - c)

        def F(a):
            y = y_fn(a); z = z_fn(a); w = w_fn(a)
            T_b = T_b_fn(y, z, w)
            # Note the sign difference vs. backwind mapping
            return a + self.a_fn(T_b, Delta_xj, BC_dict)

        # Reasonable initial guess from interior T
        k_i, _ = self.k(T_i)
        a0 = -self.a_fn(T_i, Delta_xj, BC_dict)

        # Solve scalar nonlinear equation for 'a'
        a = newton(F, x0=a0)

        y = y_fn(a); z = z_fn(a); w = w_fn(a)

        dT_i__dxj   = y / Delta_xj
        d2T_i__dxj2 = z / (Delta_xj**2)
        d3T_i__dxj3 = w / (Delta_xj**3)

        T_boundary = T_b_fn(y, z, w)
        dT_b__dxj  = dT_i__dxj + (1/2)*d2T_i__dxj2*Delta_xj + (1/8)*d3T_i__dxj3*(Delta_xj**2)

        return dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj


class Conv_BC_Stencil(Heat_Flux_BC_Stencil):
    """
    Convection (Robin) boundary condition stencil.

    Enforces -k(T_b) * (∂T/∂n)_b = h (T_b - T_inf)  ⇔  q = h (T_inf - T_b),
    which is handled by reusing the Neumann machinery via a_fn(T_b).
    """

    def __init__(self, Delta_x, Delta_y, int_stencil: Interior_FD, conv_BC, k):
        self.Delta_x = Delta_x
        self.Delta_y = Delta_y
        self.conv_BC = conv_BC     # callable/indexable returning (h, T_inf)
        self.k = k
        self.int_stencil = int_stencil

    def get_BC(self, idx, t):
        """Return convection parameters at span-wise index and time."""
        h, T_inf = self.conv_BC[idx, t]
        return {"h": h, "T_inf": T_inf}

    def a_fn(self, T_b, Delta_xj, BC_dict):
        """
        For convection: q = h (T_inf - T_b). Reuse Neumann mapping:
            a(T_b) = - Δx * q / k(T_b) = -(h (T_inf - T_b) Δx) / k(T_b)
        """
        h, T_inf = BC_dict["h"], BC_dict["T_inf"]
        k_b = self.k(T_b)[0]
        return -(h * (T_inf - T_b) * Delta_xj) / k_b
