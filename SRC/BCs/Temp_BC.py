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
    def __init__(self, mean, A, P, phi, side):
        self.const_mean = isinstance(mean, (int, float))
        self.const_A = isinstance(A, (int, float))
        self.mean = mean
        self.A = A
        self.P = P
        self.phi = phi
        self.side = side
    
    def __getitem__(self, key):
        i, t = key
        A = self.A if self.const_A else self.A[i]
        mean = self.mean if self.const_mean else self.mean[i]
        return A * np.sin((2*np.pi)/self.P * t + self.phi) + mean

