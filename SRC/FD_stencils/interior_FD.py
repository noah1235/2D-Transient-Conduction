from SRC.mat_props.thermal_cond import Thermal_Cond
from abc import ABC, abstractmethod

def CD_df__dxi(f1, f2, f3, delta_x):
    return (f1 - 2*f2 + f3) / delta_x**2

class Interior_FD(ABC):
    @abstractmethod
    def __init__(self, Nx, Ny, Delta_x, Delta_y, k: Thermal_Cond):
        pass
    
    @abstractmethod
    def stencil(self, Delta_xj, Ti, T_im1, T_ip1, T_im2, T_imp2):
        pass

    @abstractmethod
    def get_x_comp(self, i, j, T, Delta_x):
        pass
    
    @abstractmethod
    def get_y_comp(self, i, j, T, Delta_y):
        pass
    
    @abstractmethod
    def __call__(self, i, j, T, Delta_x, Delta_y):
        pass

class Interior_Central_Diff_Stencil(Interior_FD):
    def __init__(self, Nx, Ny, Delta_x, Delta_y, k: Thermal_Cond):
        self.k = k
        self.Nx = Nx
        self.Ny = Ny
        self.Delta_x = Delta_x
        self.Delta_y = Delta_y
    @staticmethod
    def CD_1D_2A(Delta_xj, T_ip1, T_im1):
        return (T_ip1-T_im1)/(2*Delta_xj)
    
    @staticmethod
    def CD_2D_2A(Delta_xj, T_i, T_ip1, T_im1):
        return (T_ip1+T_im1-2*T_i)/(Delta_xj**2)

    def stencil(self, Delta_xj, T_i, T_im1, T_ip1):
        k_i, dk_i__dT = self.k(T_i)

        return dk_i__dT * self.CD_1D_2A(Delta_xj, T_ip1, T_im1) + k_i * self.CD_2D_2A(Delta_xj, T_i, T_ip1, T_im1)

    def get_x_comp(self, i, j, T):
        return self.stencil(self.Delta_x, T[i, j], T[i-1, j], T[i+1, j])

    def get_y_comp(self, i, j, T):        
        return self.stencil(self.Delta_y, T[i, j], T[i, j-1], T[i, j+1])

    def __call__(self, i, j, T):
        return self.get_x_comp(i, j, T) + self.get_y_comp(i, j, T)

    