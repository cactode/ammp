import numpy as np

class EndMill:
    def __init__(self, N, r_c, r_s, l_c, l_s):
        self.N = N
        self.r_c = r_c
        self.r_s = r_s
        self.l_c = l_c
        self.l_s = l_s

    def unpack(self):
        return [self.N, self.r_c, self.r_s, self.l_c, self.l_s]

    def __str__(self):
        return str(self.unpack())

class MachineChar:
    def __init__(self, r_e, K_T, R_w, V_max, I_max, T_nom, f_r_max):
        self.r_e = r_e
        self.K_T = K_T
        self.R_w = R_w
        self.V_max = V_max
        self.I_max = I_max
        self.T_nom = T_nom
        self.f_r_max = f_r_max

    def unpack(self):
        return [self.r_e, self.K_T, self.R_w, self.V_max, self.I_max, self.T_nom]
        
    def __str__(self):
        return str(self.unpack())

class Conditions:
    def __init__(self, D : float, W : float, f_r : float, w : float, endmill : EndMill):
        self.D = D
        self.W = W
        self.f_r = f_r
        self.w = w       
        self.endmill = endmill     

    def unpack(self):
        return [self.D, self.W, self.f_r, self.w, self.endmill]

    def __str__(self):
        return str(self.unpack())

class Data(Conditions):
    def __init__(self, D : float, W : float, f_r : float, w : float, endmill : EndMill, Ts : np.ndarray, Fys : np.ndarray):
        super(Data, self).__init__(D, W, f_r, w, endmill)
        self.Ts = Ts
        self.Fys = Fys

    def unpack(self):
        return super(Data, self).unpack() + [self.Ts, self.Fys]

    def conditions(self):
        return Conditions(*super(Data, self).unpack())

class Prediction(Conditions):
    def __init__(self, D : float, W : float, f_r : float, w : float, endmill : EndMill, T : float, F : np.ndarray):
        super(Data, self).__init__(D, W, f_r, w, endmill)
        self.T = T
        self.F = F

    def unpack(self):
        return super(Prediction, self).unpack() + [self.T, self.F]

    def conditions(self):
        return Conditions(*super(Data, self).unpack())