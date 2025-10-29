class q_BC:
    def __init__(self, q):
        self.const_q = isinstance(q, (int, float))
        self.q = q
    
    def __getitem__(self, key):
        i, t = key
        if self.const_q:
            return self.q
        return self.q[i]