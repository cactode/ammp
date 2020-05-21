import numpy as np

class EndMill:
    def __init__(self, N, r_c, r_s, l_c, l_s):
        self.R = R
        self.N = N
        self.r_c = r_c
        self.r_s = r_s
        self.l_c = l_c
        self.l_s = l_s

class Conditions:
    def __init__(self, D : float, W : float, f_r : float, w : float, endmill : EndMill):
        self.D = D
        self.W = W
        self.f_r = f_r
        self.w = w       
        self.endmill = endmill       

class Data(Conditions):
    def __init__(self, D : float, W : float, f_r : float, w : float, endmill : EndMill, Ts : np.ndarray, Fys : np.ndarray):
        super(Data, self).__init__(D, W, f_r, w, endmill)
        self.Ts = Ts
        self.Fys = Fys

class Prediction(Conditions):
    def __init__(self, D : float, W : float, f_r : float, w : float, endmill : EndMill, T : float, F : float):
        super(Data, self).__init__(D, W, f_r, w, endmill)
        self.T = T
        self.F = F