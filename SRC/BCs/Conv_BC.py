import numpy as np
class Conv_BC:
    def __init__(self, h, T_inf):
        self.const_h = isinstance(h, (int, float))
        self.const_t_inf = isinstance(T_inf, (int, float))
        self.h = h
        self.T_inf = T_inf
    
    def __getitem__(self, key):
        i, t = key
        if self.const_h:
            h = self.h
        else:
            h = self.h[i]
        if self.const_t_inf:
            T_inf = self.T_inf
        else:
            T_inf = self.T_inf[i]
        return h, T_inf
    

class Time_Space_Conv_BC_:
    def __init__(self, h0, T0, Delta_xj, amp_h=0.2,
                 f_h=0.2, k_h=2*np.pi):
        self.h0 = h0
        self.T0 = T0
        self.amp_h = amp_h
        self.f_h = f_h
        self.k_h = k_h
        self.Delta_xj = Delta_xj

    def __call__(self, x, t):
        """
        Return (h, T_inf) at spatial coordinate x (can be scalar or array) and time t.
        """
        # Time- and space-dependent convection coefficient
        h = self.h0 * (1.0 + self.amp_h * np.sin(2*np.pi*self.f_h*t + self.k_h*x))

        return h, self.T0

    def __getitem__(self, key):
        """For backward compatibility: key = (x, t)"""
        i, t = key
        x = i * self.Delta_xj
        return self.__call__(x, t)
