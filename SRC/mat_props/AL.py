import numpy as np
from SRC.mat_props.thermal_cond import TBL_K
from SRC.mat_props.rho_and_specific_heat import Tbl_Prop
def get_AL_props():
    T_tbl = np.array([150, 200, 250, 300, 400,  600]) #kelvin
    k_tbl = np.array([250, 237, 235, 237, 240, 231]) #W/m*k
    rho_tbl = np.array([2726, 2719, 2710, 2701, 2681, 2639]) #kg/m^3
    c_tbl = 10**3 * np.array([0.683, 0.797, 0.859, 0.902, 0.949, 1.042]) #J/kg k

    k_obj = TBL_K(k_tbl, T_tbl)
    rho_obj = Tbl_Prop(T_tbl, rho_tbl)
    c_obj = Tbl_Prop(T_tbl, c_tbl)

    return k_obj, rho_obj, c_obj
