import numpy as np

class Temp_BC:
    def __init__(self, T):
        self.const_temp = isinstance(T, (int, float))
        self.T = T
    
    def __getitem__(self, key):
        i, t = key
        if self.const_temp:
            return self.T
        return self.T[i]

class Sin_Temp_BC():
    def __init__(self, mean, A, P, phi):
        self.const_mean = isinstance(mean, (int, float))
        self.const_A = isinstance(A, (int, float))
        self.mean = mean
        self.A = A
        self.P = P
        self.phi = phi

    
    def __getitem__(self, key):
        i, t = key
        A = self.A if self.const_A else self.A[i]
        mean = self.mean if self.const_mean else self.mean[i]
        return A * np.sin((2*np.pi)/self.P * t + self.phi) + mean

class T_Wave_BC():
    def __init__(self, mean, A, P, vel, Delta_xj):
        self.mean = mean
        self.A = A
        self.vel = vel
        self.P = P
        self.Delta_xj = Delta_xj
    
    def __getitem__(self, key):
        i, t = key
        x = i * self.Delta_xj
        return self.A * np.sin((2*np.pi)/self.P * x + self.vel * t) + self.mean