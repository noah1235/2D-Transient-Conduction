from SRC.FD_stencils.interior_FD import Interior_FD
from SRC.FD_stencils.BC_Stencil import BC_Stencil, Conv_BC_Stencil
from SRC.BCs.Q_internal_BC import Q
import numpy as np
from scipy.integrate import solve_ivp

class Sim_Manager:
    def __init__(self, Nx, Ny, Delta_x, Delta_y, Q_gen: Q,
                 interior_stencil: Interior_FD,
                 left_BC: BC_Stencil, top_BC: BC_Stencil, right_BC: BC_Stencil, bot_BC: BC_Stencil,
                 rho, c
                 ):
        left_BC.set_side("left")
        top_BC.set_side("top")
        right_BC.set_side("right")
        bot_BC.set_side("bot")

        self.Delta_x = Delta_x
        self.Delta_y = Delta_y
        self.Nx = Nx
        self.Ny = Ny
        self.Q_gen = Q_gen

        self.interior_stencil = interior_stencil
        self.left_BC = left_BC
        self.top_BC = top_BC
        self.right_BC = right_BC
        self.bot_BC = bot_BC
        self.rho = rho
        self.c = c


    def RHS(self, t: float, T: np.ndarray):
        #array to store approximation of the right hand side of conduction PDE
        output = np.zeros((self.Nx, self.Ny))
        crho = self.c(T) * self.rho(T)
        
        #interior nodes
        for i in range(1, self.Nx-1):
            for j in range(1, self.Ny-1):
                output[i, j] += self.interior_stencil(i, j, T)

        #left nodes
        i = 0
        for j in range(1, self.Ny-1):
            x_comp = self.left_BC.get_BC_comp(T, i, j, t)
            y_comp = self.left_BC.get_int_comp(T, i, j)
            
            output[i, j] += (x_comp + y_comp)

        #top nodes
        j = self.Ny-1
        for i in range(1, self.Nx-1):
            x_comp = self.top_BC.get_int_comp(T, i, j)
            y_comp = self.top_BC.get_BC_comp(T, i, j, t)
            output[i, j] += (x_comp + y_comp)

        #right nodes
        i = self.Nx-1
        for j in range(1, self.Ny-1):
            x_comp = self.right_BC.get_BC_comp(T, i, j, t)
            y_comp = self.right_BC.get_int_comp(T, i, j)
            output[i, j] += (x_comp + y_comp)

        #bottom nodes
        j = 0
        for i in range(1, self.Nx-1):
            x_comp = self.bot_BC.get_int_comp(T, i, j)
            y_comp = self.bot_BC.get_BC_comp(T, i, j, t)
            output[i, j] += (x_comp + y_comp)

        #top left corner
        i, j = 0, self.Ny-1
        output[i, j] += (self.left_BC.get_BC_comp(T, i, j, t) 
                         + self.top_BC.get_BC_comp(T, i, j, t)
                         )
    
        #bot left corner
        i, j = 0, 0
        output[i, j] += (self.left_BC.get_BC_comp(T, i, j, t) 
                         + self.bot_BC.get_BC_comp(T, i, j, t)
                         )

        #top right corner
        i, j = self.Nx-1, self.Ny-1
        output[i, j] += (self.right_BC.get_BC_comp(T, i, j, t) 
                         + self.top_BC.get_BC_comp(T, i, j, t)
                         )

        #bot right corner
        i, j = self.Nx-1, 0
        output[i, j] += (self.right_BC.get_BC_comp(T, i, j, t) 
                         + self.bot_BC.get_BC_comp(T, i, j, t)
                         )

        

        return output/crho + self.Q_gen(T, t)
    
    @staticmethod
    def mean_if_complete(x: np.ndarray) -> float:
        # x must be float/complex dtype to hold NaNs (your np.full(..., np.nan) already ensures this)
        return np.nan if np.isnan(x).any() else float(np.mean(x))

    def get_BC_debug(self, T, t):
        T_boundary_left = np.zeros(self.Ny)
        T_boundary_top = np.zeros(self.Nx)
        T_boundary_right = np.zeros(self.Ny)
        T_boundary_bot = np.zeros(self.Nx)

        q_boundary_left = np.zeros(self.Ny)
        q_boundary_top = np.zeros(self.Nx)
        q_boundary_right = np.zeros(self.Ny)
        q_boundary_bot = np.zeros(self.Nx)

        conv_boundary_left = np.full(self.Ny, np.nan)
        conv_boundary_top = np.full(self.Nx, np.nan)
        conv_boundary_right = np.full(self.Ny, np.nan)
        conv_boundary_bot = np.full(self.Nx, np.nan)

        #left nodes
        i = 0
        for j in range(self.Ny):
            self.left_BC.get_BC_comp(T, i, j, t, T_boundary_left, q_boundary_left, conv_boundary_left)

        #top nodes
        j = self.Ny-1
        for i in range(self.Nx):
            self.top_BC.get_BC_comp(T, i, j, t, T_boundary_top, q_boundary_top, conv_boundary_top)

        #right nodes
        i = self.Nx-1
        for j in range(self.Ny):
            self.right_BC.get_BC_comp(T, i, j, t, T_boundary_right, q_boundary_right, conv_boundary_right)

        #bottom nodes
        j = 0
        for i in range(self.Nx):
            self.bot_BC.get_BC_comp(T, i, j, t, T_boundary_bot, q_boundary_bot, conv_boundary_bot)


        T_left_mean = np.mean(T_boundary_left)
        T_right_mean = np.mean(T_boundary_right)
        T_top_mean = np.mean(T_boundary_top)
        T_bot_mean = np.mean(T_boundary_bot)

        q_left_mean = np.mean(q_boundary_left)
        q_top_mean = np.mean(q_boundary_top)
        q_right_mean = np.mean(q_boundary_right)
        q_bot_mean = np.mean(q_boundary_bot)

        means = [self.mean_if_complete(a) for a in
                (conv_boundary_left, conv_boundary_top, conv_boundary_right, conv_boundary_bot)]

        conv_left_mean, conv_top_mean, conv_right_mean, conv_bot_mean = means

        BC_debug = np.array(
            [
                [T_left_mean, q_left_mean, conv_left_mean],
                [T_top_mean, q_top_mean, conv_top_mean],
                [T_right_mean, q_right_mean, conv_right_mean],
                [T_bot_mean, q_bot_mean, conv_bot_mean]
            ]
        )

        return BC_debug

    def _rhs_flat(self, t: float, y_flat: np.ndarray) -> np.ndarray:
        """solve_ivp-compatible RHS that maps (t, y_flat) -> y_flat'."""
        T = y_flat.reshape(self.Nx, self.Ny)
        dTdt = self.RHS(t, T)  
        return dTdt.ravel(order="C") 

    def __call__(self, T0, t_span, t_eval, method="RK45"):
        assert T0.shape == (self.Nx, self.Ny)

        y0 = T0.ravel(order="C")
        sol = solve_ivp(fun=self._rhs_flat,
                        t_span=t_span,
                        t_eval=t_eval,
                        method=method,
                        y0=y0
                        )
        trj = sol.y.T.reshape(-1, self.Nx, self.Ny)

        BC_debug = np.zeros((trj.shape[0], 4, 3))
        for i in range(trj.shape[0]):
            BC_debug[i] = self.get_BC_debug(trj[i], t_eval[i])


        return trj, BC_debug

             
        



