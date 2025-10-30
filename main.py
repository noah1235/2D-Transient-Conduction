from SRC.mat_props.rho_and_specific_heat import Const_Prop
from SRC.mat_props.thermal_cond import Const_k
from SRC.FD_stencils.interior_FD import Interior_Central_Diff_Stencil
from SRC.Sim_Manager import Sim_Manager
from SRC.BCs.Temp_BC import Temp_BC, Sin_Temp_BC, T_Wave_BC
from SRC.BCs.heat_flux_BC import q_BC
from SRC.BCs.Conv_BC import Conv_BC, Time_Space_Conv_BC_
from SRC.FD_stencils.BC_Stencil import Temp_BC_Stencil, Conv_BC_Stencil, Heat_Flux_BC_Stencil
import matplotlib.pyplot as plt
from SRC.BC_IC_gen.Temp_IC_gen import generate_random_temp_BC
from SRC.plotting.ts_ani import animate_trj
from SRC.plotting.BC_plotting import plot_q_residual_T_BC
import numpy as np
from SRC.BCs.Q_internal_BC import Q_gen_cent_square, QSpatioTemporalArt
from SRC.mat_props.AL import get_AL_props

def main():
    Lx = 0.3 #m
    Ly = 0.3 #m
    Nx = 20
    Ny = 20
    Delta_x = Lx / Nx
    Delta_y = Ly / Ny
    k, rho, c = get_AL_props()

    Q_gen = Q_gen_cent_square(0, Nx, Ny, px=.25, py=.25)
    #Q_gen = QSpatioTemporalArt(Nx, Ny, A_blob=10)

    int_stencil = Interior_Central_Diff_Stencil(Nx, Ny, Delta_x, Delta_y, k)


    left_BC = Temp_BC_Stencil(Delta_x, Delta_y, 
                            int_stencil,
                            Temp_BC(300),
                            #Sin_Temp_BC(mean=np.ones(Ny)*300, A=50, P=5, phi=np.pi/2, side="left"),
                            #T_Wave_BC(mean=300, A=100, P=.08, vel=.2, Delta_xj=Delta_y),
                            k
                            )
    top_BC = Temp_BC_Stencil(Delta_x, Delta_y, 
                            int_stencil,
                            Temp_BC(300),
                            #T_Wave_BC(mean=300, A=100, P=.08, vel=.2, Delta_xj=Delta_x),
                            #Sin_Temp_BC(mean=300, A=10, P=1, phi=.35, side="top"),
                            k
                            )
    right_BC = Temp_BC_Stencil(Delta_x, Delta_y, 
                            int_stencil,
                            Temp_BC(300),
                            #Sin_Temp_BC(mean=300, A=15, P=5, phi=.0, side="right"),
                            #T_Wave_BC(mean=300, A=100, P=.08, vel=-.2, Delta_xj=Delta_y),
                            k
                            )
    bot_BC = Temp_BC_Stencil(Delta_x, Delta_y, 
                            int_stencil,
                            Temp_BC(300),
                            #T_Wave_BC(mean=300, A=100, P=.08, vel=-.2, Delta_xj=Delta_x),
                            k
                            )
    right_BC = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, Conv_BC(100, 400), k)
    left_BC = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, Conv_BC(100, 400), k)
    top_BC = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, Conv_BC(100, 400), k)
    bot_BC = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, Conv_BC(100, 400), k)
    #bot_BC = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, Time_Space_Conv_BC_(h0=100, T0=400, Delta_xj=Delta_x, amp_h=10, k_h=10), k)
    #top_BC = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, Time_Space_Conv_BC_(h0=100, T0=200, Delta_xj=Delta_x, amp_h=10, k_h=-10), k)

    #right_BC = Heat_Flux_BC_Stencil(Delta_x, Delta_y, int_stencil, q_BC(10000), k)
    #left_BC = Heat_Flux_BC_Stencil(Delta_x, Delta_y, int_stencil, q_BC(10000), k)
    #top_BC = Heat_Flux_BC_Stencil(Delta_x, Delta_y, int_stencil, q_BC(10000), k)
    #bot_BC = Heat_Flux_BC_Stencil(Delta_x, Delta_y, int_stencil, q_BC(10000), k)


    sim_manager = Sim_Manager(Nx, Ny, Delta_x, Delta_y, Q_gen,
                              int_stencil, left_BC, top_BC, right_BC, bot_BC,
                              rho, c)

    time = 50
    dt = .5
    T0 = np.ones((Nx, Ny)) * 200
    t_eval = np.arange(0, time, dt)
    trj, BC_debug = sim_manager(T0, (0, time), t_eval, method="Radau")

    plot_q_residual_T_BC(BC_debug[10:], t_eval[10:], boundaries=("bottom",), save_path="BC_plot.svg")
    animate_trj(trj, extent=(0, Lx, 0, Ly),
                dt=dt, fps=5, save="test.mp4")



main()