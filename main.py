from SRC.mat_props.rho_and_specific_heat import Const_Prop
from SRC.mat_props.thermal_cond import Const_k
from SRC.FD_stencils.interior_FD import Interior_Central_Diff_Stencil
from SRC.Sim_Manager import Sim_Manager
from SRC.BCs.Temp_BC import Temp_BC, Sin_Temp_BC
from SRC.BCs.heat_flux_BC import q_BC
from SRC.BCs.Conv_BC import Conv_BC
from SRC.FD_stencils.BC_Stencil import Temp_BC_Stencil, Conv_BC_Stencil, Heat_Flux_BC_Stencil
import matplotlib.pyplot as plt
from SRC.BC_IC_gen.Temp_IC_gen import generate_random_temp_BC
from SRC.plotting.ts_ani import animate_trj
from SRC.plotting.BC_plotting import plot_q_residual_T_BC
import numpy as np
from SRC.BCs.Q_internal_BC import Q_gen_cent_square
from SRC.mat_props.AL import get_AL_props

def main():
    Lx = 0.3 #m
    Ly = 0.3 #m
    Nx = 40
    Ny = 40
    Delta_x = Lx / Nx
    Delta_y = Ly / Ny
    k, rho, c = get_AL_props()

    Q_gen = Q_gen_cent_square(0, Nx, Ny, px=.75, py=.1)

    int_stencil = Interior_Central_Diff_Stencil(Nx, Ny, Delta_x, Delta_y, k)


    left_BC = Temp_BC_Stencil(Delta_x, Delta_y, 
                            int_stencil,
                            Temp_BC(300),
                            #Sin_Temp_BC(mean=300, A=50, P=5, phi=np.pi/2, side="left"),
                            k
                            )
    top_BC = Temp_BC_Stencil(Delta_x, Delta_y, 
                            int_stencil,
                            Temp_BC(300),
                            #Sin_Temp_BC(mean=300, A=10, P=1, phi=.35, side="top"),
                            k
                            )
    right_BC = Temp_BC_Stencil(Delta_x, Delta_y, 
                            int_stencil,
                            Temp_BC(300),
                            #Sin_Temp_BC(mean=300, A=15, P=5, phi=.0, side="right"),
                            k
                            )
    bot_BC = Temp_BC_Stencil(Delta_x, Delta_y, 
                            int_stencil,
                            Temp_BC(300),
                            k
                            )
    #right_BC = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, Conv_BC(100, 320), k)
    #left_BC = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, Conv_BC(100, 320), k)
    #top_BC = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, k)

    #right_BC = Heat_Flux_BC_Stencil(Delta_x, Delta_y, int_stencil, q_BC(0), k)
    #top_BC = Heat_Flux_BC_Stencil(Delta_x, Delta_y, int_stencil, q_BC(0), k)
    #bot_BC = Heat_Flux_BC_Stencil(Delta_x, Delta_y, int_stencil, q_BC(0), k)


    sim_manager = Sim_Manager(Nx, Ny, Delta_x, Delta_y, Q_gen,
                              int_stencil, left_BC, top_BC, right_BC, bot_BC,
                              rho, c)

    time = 40
    dt = 1
    T0 = np.ones((Nx, Ny)) * 200
    t_eval = np.arange(0, time, dt)
    trj, BC_debug = sim_manager(T0, (0, time), t_eval, method="RK45")

    plot_q_residual_T_BC(BC_debug[10:], t_eval[10:], boundaries=("left",))
    animate_trj(trj, extent=(0, Lx, 0, Ly),
                dt=dt, fps=5, save="const_temp.mp4")



main()