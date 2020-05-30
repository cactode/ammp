import numpy as np
from json import dumps


class EndMill:
    """
    Represents an endmill

    Args:
        N: Number of flutes
        r_c: Radius of cutter
        r_s: Radius of shank
        l_c: Length of cutter
        l_s: Length of shank (not including cutter)
    """

    def __init__(self, N, r_c, r_s, l_c, l_s):
        self.N = N
        self.r_c = r_c
        self.r_s = r_s
        self.l_c = l_c
        self.l_s = l_s

    def unpack(self):
        return [self.N, self.r_c, self.r_s, self.l_c, self.l_s]

    def __str__(self):
        info = {"N": self.N,
                "r_c": self.r_c,
                "r_s": self.r_s,
                "l_c": self.l_c,
                "l_s": self.l_s}
        return "Endmill: (" + ", ".join([(name + ": " + str(value)) for name, value in info.items()]) + ")"


class MachineChar:
    """ 
    Represents a machine's unique performance characteristics.

    Args:
        r_e: Ratio from motor to spindle
        K_T: Torque constant of motor
        R_w: Winding resistance of motor
        V_max: Max bus voltage for motor
        I_max: Max current draw of motor
        T_nom: Nominal friction torque of spindle
    """

    def __init__(self, r_e, K_T, R_w, V_max, I_max, T_nom, f_r_max):
        self.r_e = r_e
        self.K_T = K_T
        self.R_w = R_w
        self.V_max = V_max
        self.I_max = I_max
        self.T_nom = T_nom
        self.f_r_max = f_r_max

    def unpack(self):
        return [self.r_e, self.K_T, self.R_w, self.V_max, self.I_max, self.T_nom, self.f_r_max]

    def __str__(self):
        return str(self.unpack())


class Conditions:
    """
    Represents cutting conditions.

    Args:
        D: Depth of cut
        W: Width of cut
        f_r: Feed rate
        w: Cutter rotational rate
        endmill: An endmill object
    """

    def __init__(self, D: float, W: float, f_r: float, w: float, endmill: EndMill):
        self.D = D
        self.W = W
        self.f_r = f_r
        self.w = w
        self.endmill = endmill

    def unpack(self):
        return [self.D, self.W, self.f_r, self.w, self.endmill]

    def conditions(self):
        return self

    def __str__(self):
        info = {"D": self.D,
                "W": self.W,
                "f_r": self.f_r,
                "w": self.w,
                "endmill": str(self.endmill)}
        return "Conditions: (" + ", ".join([(name + ": " + str(value)) for name, value in info.items()]) + ")"


class Data(Conditions):
    """
    Represents data returned from a cutting operation.

    Args:
        D: Depth of cut
        W: Width of cut
        f_r: Feed rate
        w: Cutter rotational rate
        endmill: An endmill object
        Ts: A (2 x N) array of measured cutting torques. Ts[0, :] is times, Ts[1, :] is torques.
        Fys: A (2 x N) array of the same format representing uniaxial cutting forces.
    """

    def __init__(self, D: float, W: float, f_r: float, w: float, endmill: EndMill, Ts: np.ndarray, Fys: np.ndarray):
        super(Data, self).__init__(D, W, f_r, w, endmill)
        self.Ts = Ts
        self.Fys = Fys

    def unpack(self):
        return super(Data, self).unpack() + [self.Ts, self.Fys]

    def conditions(self):
        """
        Pulls conditions from a Data object.
        """
        return Conditions(*super().unpack())

    def __str__(self):
        info = {"D": self.D,
                "W": self.W,
                "f_r": self.f_r,
                "w": self.w,
                "endmill": str(self.endmill),
                "Ts average": np.median(self.Ts[1:]),
                "Fys average": np.median(self.Fys[1:])}
        return "Data: (" + ", ".join([(name + ": " + str(value)) for name, value in info.items()]) + ")"


class Prediction(Conditions):
    """
    Represents data returned from a cutting operation.

    Args:
        D: Depth of cut
        W: Width of cut
        f_r: Feed rate
        w: Cutter rotational rate
        endmill: An endmill object
        T: Predicted cutting torque
        F: Predicted cutting forces
    """

    def __init__(self, D: float, W: float, f_r: float, w: float, endmill: EndMill, T: float, F: np.ndarray):
        super().__init__(D, W, f_r, w, endmill)
        self.T = T
        self.F = F

    def unpack(self):
        return super(Prediction, self).unpack() + [self.T, self.F]

    def conditions(self):
        """
        Pulls conditions from a Data object.
        """
        return Conditions(*super().unpack())

    def __str__(self):
        info = {"D": self.D,
                "W": self.W,
                "f_r": self.f_r,
                "w": self.w,
                "endmill": str(self.endmill),
                "T": self.T,
                "F": self.F}

        return "Prediction: (" + ", ".join([(name + ": " + str(value)) for name, value in info.items()]) + ")"
