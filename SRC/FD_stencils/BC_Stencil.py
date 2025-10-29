
from abc import ABC, abstractmethod
from SRC.BCs.Temp_BC import Temp_BC
from SRC.FD_stencils.interior_FD import Interior_FD
from scipy.optimize import root_scalar
from scipy.optimize import newton
import numpy as np

class BC_Stencil(ABC):

    def set_side(self, side):
        self.side = side

    @abstractmethod
    def upwind_BC_ctrl(self, T, i, j, t):
        pass

    @abstractmethod
    def backwind_BC_ctrl(self, T, i, j, t):
        pass

    @abstractmethod
    def backwind_BC_stencil(Delta_xj, T_i, T_ip1, T_ip2, BC_dict):
        pass

    @abstractmethod
    def upwind_BC_stencil(Delta_xj, T_i, T_im1, T_im2, BC_dict):
        pass

    @abstractmethod
    def get_BC(self, i, t):
        pass 

    def get_BC_comp(self, T, i, j, t, T_boundary_list=None, q_boundary_list=None, conv_boundary_list=None):
        if self.side == "left" or self.side == "bot":
            comp, T_boundary, boundary_heat_flux = self.backwind_BC_ctrl(T, i, j, t)
        elif self.side == "right" or self.side == "top":
            comp, T_boundary, boundary_heat_flux = self.upwind_BC_ctrl(T, i, j, t)
        BC_debug = (T_boundary_list is not None) and (q_boundary_list is not None)
        if BC_debug:
            if self.side == "left" or self.side == "right":
                T_boundary_list[j] = T_boundary
                q_boundary_list[j] = boundary_heat_flux
                if isinstance(self, Conv_BC_Stencil):
                    h, T_inf = self.conv_BC[j, t]
                    conv_boundary_list[j] = h * (T_boundary - T_inf)
            else:
                T_boundary_list[i] = T_boundary
                q_boundary_list[i] = boundary_heat_flux
                if isinstance(self, Conv_BC_Stencil):
                    h, T_inf = self.conv_BC[i, t]
                    conv_boundary_list[i] = h * (T_boundary - T_inf)

        return comp
        
    def get_int_comp(self, T, i, j):
        if self.side == "left":
            return self.int_stencil.get_y_comp(i, j, T)
        elif self.side == "bot":
            return self.int_stencil.get_x_comp(i, j, T)
        elif self.side == "right":
            return self.int_stencil.get_y_comp(i, j, T)
        elif self.side == "top":
            return self.int_stencil.get_x_comp(i, j, T)
        
    def upwind_BC_ctrl(self, T, i, j, t):
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

        k_i, dk_i__dT = self.k(T_i)
        dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj = self.upwind_BC_stencil(Delta_xj, T_i, T_im1, T_im2, BC_dict)
        k_boundary = self.k(T_boundary)[0]
        boundary_heat_flux = -k_boundary*dT_b__dxj

        return dk_i__dT * dT_i__dxj + k_i * d2T_i__dxj2, T_boundary, boundary_heat_flux

    def backwind_BC_ctrl(self, T, i, j, t):
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
        dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj = self.backwind_BC_stencil(Delta_xj, T_i, T_ip1, T_ip2, BC_dict)
        k_boundary = self.k(T_boundary)[0]
        boundary_heat_flux = -k_boundary*dT_b__dxj


        return dk_i__dT * dT_i__dxj + k_i * d2T_i__dxj2, T_boundary, boundary_heat_flux
    
class Temp_BC_Stencil(BC_Stencil):
    def __init__(self, Delta_x, Delta_y, int_stencil: Interior_FD, T_BC: Temp_BC, k):
        self.Delta_x = Delta_x
        self.Delta_y = Delta_y
        self.T_BC = T_BC
        self.k = k
        self.int_stencil = int_stencil

        self.T_boundary_debug = np.zeros_like(T_BC)
        self.q_boundary_debug = np.zeros_like(T_BC)
    
    def get_BC(self, idx, t):
        return {"T_BC": self.T_BC[idx, t]}

    @staticmethod
    def upwind_BC_stencil(Delta_xj, T_i, T_im1, T_im2, BC_dict):
        T_bc = BC_dict["T_BC"]
        a = T_bc - T_i
        b = T_im1 - T_i
        c = T_im2 - T_i
        y = (
                (16/15)*a
                -(2/3)*b
                +(1/10)*c
            
        )
        z = (
                (16/5)*a
                +2*b
                -(1/5)*c
            
        )
        w = (
                (16/5)*a
                +4*b
                -(6/5)*c
            
        )
        dT_i__dxj = y/Delta_xj
        d2T_i__dxj2 = z/Delta_xj**2
        d3T_i__dxj3 = w/Delta_xj**3
        
        T_boundary = T_i +(1/2)*y + (1/8)*z + (1/48) * w
        dT_b__dxj = dT_i__dxj + (1/2) * d2T_i__dxj2 * Delta_xj + (1/8) * d3T_i__dxj3*Delta_xj**2

        return dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj

    @staticmethod
    def backwind_BC_stencil(Delta_xj, T_i, T_ip1, T_ip2, BC_dict):
        T_bc = BC_dict["T_BC"]

        a = T_bc - T_i
        b = T_ip1 - T_i
        c = T_ip2 - T_i
        y = (
                -(16/15)*a
                +(2/3)*(b)
                -(1/10)*(c)  
        )

        z = (
                (16/5)*a
                +2*b
                -(1/5)*c
            
        )
        w = (
                -(16/5)*a
                -4*b
                +(6/5)*c
            
        )
        dT_i__dxj = y/Delta_xj
        d2T_i__dxj2 = z / Delta_xj**2
        d3T_i__dxj3 = w / Delta_xj**3

        #computing boundary values
        T_boundary = T_i -(1/2)*y + (1/8)*z - (1/48) * w
        dT_b__dxj = dT_i__dxj - (1/2) * d2T_i__dxj2 * Delta_xj + (1/8) * d3T_i__dxj3*Delta_xj**2

        return dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj


class Heat_Flux_BC_Stencil(BC_Stencil):
    def __init__(self, Delta_x, Delta_y, int_stencil: Interior_FD, q_BC, k):
        self.Delta_x = Delta_x
        self.Delta_y = Delta_y
        self.q_BC = q_BC
        self.k = k
        self.int_stencil = int_stencil

    def get_BC(self, idx, t):
        return {"q": self.q_BC[idx, t]}
    
    def a_fn(self, T_b, Delta_xj, BC_dict):
        q = BC_dict["q"]
        k_b = self.k(T_b)[0]
        return -((Delta_xj * q)/k_b)

    def backwind_BC_stencil(self, Delta_xj, T_i, T_ip1, T_ip2, BC_dict):
        b = T_ip1 - T_i
        c = T_ip2 - T_i
        T_b_fn = lambda y, z, w: T_i - (1/2) * y +(1/8)*z - (1/48)*w

        y_fn = lambda a: (16*a + 44*b - 7*c)/46
        z_fn = lambda a: (-24*a + 26*b - c)/23
        w_fn = lambda a: (24/23) * (a - 3*b + c)


        def F(a):
            y = y_fn(a)
            z = z_fn(a)
            w = w_fn(a)
            T_b = T_b_fn(y, z, w)
            return a - self.a_fn(T_b, Delta_xj, BC_dict)

        a0 = self.a_fn(T_i, Delta_xj, BC_dict)
        a = newton(F, x0=a0)

        y = y_fn(a)
        z = z_fn(a)
        w = w_fn(a)


        dT_i__dxj = y / Delta_xj
        d2T_i__dxj2 = z / Delta_xj**2
        d3T_i__dxj3 = w / Delta_xj**3

        T_boundary = T_b_fn(y, z, w)
        dT_b__dxj = dT_i__dxj - (1/2) * d2T_i__dxj2 * Delta_xj + (1/8) * d3T_i__dxj3*Delta_xj**2

        return dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj

    def upwind_BC_stencil(self, Delta_xj, T_i, T_im1, T_im2, BC_dict):

        b = T_im1 - T_i
        c = T_im2 - T_i
        T_b_fn = lambda y, z, w: T_i + (1/2) * y + (1/8)*z + (1/48)*w

        y_fn = lambda a: (16*a - 44*b + 7*c)/46
        z_fn = lambda a: (24*a + 26*b - c)/23
        w_fn = lambda a: (24/23) * (a + 3*b - c)


        def F(a):
            y = y_fn(a)
            z = z_fn(a)
            w = w_fn(a)
            T_b = T_b_fn(y, z, w)
            return a - self.a_fn(T_b, Delta_xj, BC_dict)

        k_i, _ = self.k(T_i)
        a0 = self.a_fn(T_i, Delta_xj, BC_dict)
        
        #a = a0
        a = newton(F, x0=a0)

        y = y_fn(a)
        z = z_fn(a)
        w = w_fn(a)


        dT_i__dxj = y / Delta_xj
        d2T_i__dxj2 = z / Delta_xj**2
        d3T_i__dxj3 = w / Delta_xj**3

        T_boundary = T_b_fn(y, z, w)
        dT_b__dxj = dT_i__dxj + (1/2) * d2T_i__dxj2 * Delta_xj + (1/8) * d3T_i__dxj3*Delta_xj**2

        return dT_i__dxj, d2T_i__dxj2, T_boundary, dT_b__dxj
    

class Conv_BC_Stencil(Heat_Flux_BC_Stencil):
    def __init__(self, Delta_x, Delta_y, int_stencil: Interior_FD, conv_BC, k):
        self.Delta_x = Delta_x
        self.Delta_y = Delta_y
        self.conv_BC = conv_BC
        self.k = k
        self.int_stencil = int_stencil

    def get_BC(self, idx, t):
        h, T_inf = self.conv_BC[idx, t]
        return {"h": h, "T_inf": T_inf}

    def a_fn(self, T_b, Delta_xj, BC_dict):
        h, T_inf = BC_dict["h"], BC_dict["T_inf"]
        k_b = self.k(T_b)[0]
        return (h*(T_inf-T_b)*Delta_xj) / (k_b)
