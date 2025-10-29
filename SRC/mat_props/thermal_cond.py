from abc import ABC, abstractmethod
import numpy as np
from SRC.mat_props.tbl_linear_interp import interp_tabl
class Thermal_Cond(ABC):
    @abstractmethod
    def __call__(self, T_i):
        pass

class Const_k(Thermal_Cond):
    def __init__(self, k):
        self.k = k
    def __call__(self, T_i):
        return self.k, 0

class TBL_K(Thermal_Cond):
    def __init__(self, k_tbl, T_tbl):
        sort_idx = np.argsort(T_tbl)
        self.k_tbl = k_tbl[sort_idx]
        self.T_tbl = T_tbl[sort_idx]

        
    def __call__(self, T_i):
        return interp_tabl(self.T_tbl, self.k_tbl, T_i)