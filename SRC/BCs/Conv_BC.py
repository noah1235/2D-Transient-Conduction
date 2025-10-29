
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