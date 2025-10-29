from abc import ABC, abstractmethod
import numpy as np
from SRC.mat_props.tbl_linear_interp import interp_tabl
class Material_Prop(ABC):
    @abstractmethod
    def __call__(self, T_i):
        pass
    
class Const_Prop(Material_Prop):
    def __init__(self, p):
        self.p = p

    def __call__(self, T_i):
        if isinstance(T_i, np.ndarray):
            return np.ones_like(T_i) * self.p
        else:
            return self.p
        
class Tbl_Prop(Material_Prop):
    def __init__(self, x_tbl, y_tbl):
        sort_idx = np.argsort(x_tbl)
        self.y_tbl = y_tbl[sort_idx]
        self.x_tbl = x_tbl[sort_idx]

        
    def __call__(self, x):
        return interp_tabl(self.x_tbl, self.y_tbl, x)[0]