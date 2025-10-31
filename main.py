# --- Material & stencil imports ---
from SRC.mat_props.rho_and_specific_heat import Const_Prop
from SRC.mat_props.thermal_cond import Const_k
from SRC.FD_stencils.interior_FD import Interior_Central_Diff_Stencil

# --- Simulation manager ---
from SRC.Sim_Manager import Sim_Manager

# --- Boundary condition models (temperature / heat-flux / convection) ---
from SRC.BCs.Temp_BC import Const_Temp_BC, Sin_Temp_BC, T_Wave_BC
from SRC.BCs.heat_flux_BC import Const_Heat_Flux_BC
from SRC.BCs.Conv_BC import Const_Conv_BC, Time_Space_Conv_BC

# --- Boundary-condition stencils wrapping the models above ---
from SRC.FD_stencils.BC_Stencil import (
    Temp_BC_Stencil,
    Conv_BC_Stencil,
    Heat_Flux_BC_Stencil,
)

# --- IC/plotting/utils ---
from SRC.plotting.ts_ani import animate_trj
from SRC.plotting.BC_plotting import plot_q_residual_T_BC
import numpy as np
from SRC.BCs.Q_internal_BC import Q_gen_cent_square, QSpatioTemporalArt
from SRC.mat_props.AL import get_AL_props


def main():
    # ---------------------------
    # Geometry & grid parameters
    # ---------------------------
    Lx = 0.3  # [m] domain length in x
    Ly = 0.3  # [m] domain length in y
    Nx = 20   # number of cells in x
    Ny = 20   # number of cells in y
    Delta_x = Lx / Nx
    Delta_y = Ly / Ny

    # ---------------------------
    # Time grid & ICs
    # ---------------------------
    time = 50           # [s] total time
    dt = 0.5            # [s] output step for t_eval (solver may substep internally)
    T0 = np.ones((Nx, Ny)) * 200  # [K] uniform initial temperature field
    t_eval = np.arange(0, time, dt)
    # ---------------------------
    # Numerical integration algorithim algorithin
    # ---------------------------
    method = "Radau"

    # ---------------------------
    # Material properties (Al)
    # ---------------------------
    k, rho, c = get_AL_props()

    # -----------------------------------------------
    # Internal heat generation (choose ONE; default:)
    # -----------------------------------------------
    # Centered square source: active in central region
    Q_gen = Q_gen_cent_square(0, Nx, Ny, px=0.25, py=0.25)
    # Alternative: spatio-temporal artificial source
    #Q_gen = QSpatioTemporalArt(Nx, Ny, A_blob=10)

    # -----------------------------------------------
    # Interior finite-difference stencil (central diff)
    # -----------------------------------------------
    int_stencil = Interior_Central_Diff_Stencil(Nx, Ny, Delta_x, Delta_y, k)

    # ----------------------------------------------------------------
    # BOUNDARY CONDITIONS: Dirichlet (isothermal)
    # ----------------------------------------------------------------
    left_BC = Temp_BC_Stencil(
        Delta_x, Delta_y, int_stencil, 
        Const_Temp_BC(300), 
        #Sin_Temp_BC(mean=np.ones(Ny)*300, A=50, P=5, phi=np.pi/2),
        #T_Wave_BC(mean=300, A=100, P=0.08, vel=0.2, Delta_xj=Delta_y),
        k)

    top_BC = Temp_BC_Stencil(
        Delta_x, Delta_y, int_stencil, 
        Const_Temp_BC(300), 
        #Sin_Temp_BC(mean=np.ones(Ny)*300, A=50, P=5, phi=np.pi/2),
        #T_Wave_BC(mean=300, A=100, P=0.08, vel=0.2, Delta_xj=Delta_y),
        k)

    right_BC = Temp_BC_Stencil(
        Delta_x, Delta_y, int_stencil, 
        Const_Temp_BC(300), 
        #Sin_Temp_BC(mean=np.ones(Ny)*300, A=50, P=5, phi=np.pi/2),
        #T_Wave_BC(mean=300, A=100, P=0.08, vel=0.2, Delta_xj=Delta_y),
        k)

    bot_BC = Temp_BC_Stencil(
        Delta_x, Delta_y, int_stencil, 
        Const_Temp_BC(300), 
        #Sin_Temp_BC(mean=np.ones(Ny)*300, A=50, P=5, phi=np.pi/2),
        #T_Wave_BC(mean=300, A=100, P=0.08, vel=0.2, Delta_xj=Delta_y),
        k)

    # ----------------------------------------------------------------
    # BOUNDARY CONDITIONS: Convection
    # ----------------------------------------------------------------
    left_BC  = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, 
                               Const_Conv_BC(100, 400), 
                               #Time_Space_Conv_BC(h0=100, T0=400, Delta_xj=Delta_x, amp_h=10, k_h=10),
                               k)
    top_BC   = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, 
                               Const_Conv_BC(100, 400), 
                               #Time_Space_Conv_BC(h0=100, T0=400, Delta_xj=Delta_x, amp_h=10, k_h=10),
                               k)
    right_BC = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, 
                               Const_Conv_BC(100, 400), 
                               #Time_Space_Conv_BC(h0=100, T0=400, Delta_xj=Delta_x, amp_h=10, k_h=10),
                               k)
    bot_BC   = Conv_BC_Stencil(Delta_x, Delta_y, int_stencil, 
                               Const_Conv_BC(100, 400), 
                               #Time_Space_Conv_BC(h0=100, T0=400, Delta_xj=Delta_x, amp_h=10, k_h=10),
                               k)

    # ----------------------------------------------------------------
    # BOUNDARY CONDITIONS: Heat flux
    # ----------------------------------------------------------------
    left_BC = Heat_Flux_BC_Stencil(Delta_x, Delta_y, int_stencil, 
                                    Const_Heat_Flux_BC(10000), 
                                    k)
    top_BC  = Heat_Flux_BC_Stencil(Delta_x, Delta_y, int_stencil, 
                                    Const_Heat_Flux_BC(10000), 
                                    k)
    right_BC   = Heat_Flux_BC_Stencil(Delta_x, Delta_y, int_stencil, 
                                    Const_Heat_Flux_BC(10000), 
                                    k)
    bot_BC   = Heat_Flux_BC_Stencil(Delta_x, Delta_y, int_stencil, 
                                    Const_Heat_Flux_BC(10000), 
                                    k)

    # ---------------------------
    # Simulation setup
    # ---------------------------
    sim_manager = Sim_Manager(
        Nx, Ny, Delta_x, Delta_y, Q_gen,
        int_stencil, left_BC, top_BC, right_BC, bot_BC,
        rho, c
    )


    # ---------------------------
    # Run transient simulation
    # ---------------------------
    # method="Radau" matches the original; returns trajectory and BC diagnostics
    trj, BC_debug = sim_manager(T0, (0, time), t_eval, method=method)

    # ---------------------------
    # Diagnostics / plotting
    # ---------------------------
    # Plot BC residuals/diagnostics
    plot_q_residual_T_BC(
        BC_debug[10:],      
        t_eval[10:],
        boundaries=("bottom",),
        save_path="BC_plot.svg"
    )

    # Save an animation of the temperature trajectory as MP4
    animate_trj(
        trj,
        extent=(0, Lx, 0, Ly),
        dt=dt,
        fps=5,
        save="test.mp4"
    )


if __name__ == "__main__":
     main()

