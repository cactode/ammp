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
        """
        Returns variables in the order:
        N, r_c, r_s, l_c, l_s
        """
        return [self.N, self.r_c, self.r_s, self.l_c, self.l_s]

    def __str__(self):
        info = {"N": "{:.5f}".format(self.N),
                "r_c": "{:.3f}".format(self.r_c),
                "r_s": "{:.3f}".format(self.r_s),
                "l_c": "{:.3f}".format(self.l_c),
                "l_s": "{:.3f}".format(self.l_s)}
        return "Endmill: (" + ", ".join([(name + ": " + value) for name, value in info.items()]) + ")"


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

    def __init__(self, r_e, K_T, R_w, V_max, I_max, T_nom, f_r_max, K_machine, D_a):
        self.r_e = r_e
        self.K_T = K_T
        self.R_w = R_w
        self.V_max = V_max
        self.I_max = I_max
        self.T_nom = T_nom
        self.f_r_max = f_r_max
        self.K_machine = K_machine
        self.D_a = D_a

    def unpack(self):
        """
        Returns variables in the order:
        r_e, K_T, R_w, V_max, I_max, T_nom, f_r_max, K_machine, D_a
        """
        return [self.r_e, self.K_T, self.R_w, self.V_max, self.I_max, self.T_nom, self.f_r_max, self.K_machine, self.D_a]

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
        """
        Returns variables in the order:
        D, W, f_r, w, endmill
        """
        return [self.D, self.W, self.f_r, self.w, self.endmill]

    def compromise(self, other, pref_to_other : float):
        """
        Returns a linear interpolation between this condition and another (as decided by preference_to_other).
        Endmill doesn't change.

        Args:
            other: Another condition object.
            pref_to_other: How much other will be weighted relative to this condition.
        """
        self_conditions = self.unpack()[:-1]
        other_conditions = other.unpack()[:-1]
        between = [s * (1-pref_to_other) + o * pref_to_other for s, o in zip(self_conditions, other_conditions)]
        return Conditions(*between, self.endmill)

    def conditions(self):
        return self

    def __str__(self):
        info = {"D": "{:.5f}".format(self.D),
                "W": "{:.5f}".format(self.W),
                "f_r": "{:.5f}".format(self.f_r),
                "w": "{:.5f}".format(self.w),
                "endmill": str(self.endmill)}
        return "Conditions: (" + ", ".join([(name + ": " + value) for name, value in info.items()]) + ")"


class Data(Conditions):
    """
    Represents data returned from a cutting operation.

    Args:
        D: Depth of cut
        W: Width of cut
        f_r: Feed rate
        w: Cutter rotational rate
        endmill: An endmill object
        Ts: A (N x 2) array of measured cutting torques. Ts[:, 0] is times, Ts[:, 1] is torques.
        Fys: A (N x 2) array of the same format representing uniaxial cutting forces.
    """

    def __init__(self, D: float, W: float, f_r: float, w: float, endmill: EndMill, Ts: np.ndarray, Fys: np.ndarray):
        super(Data, self).__init__(D, W, f_r, w, endmill)
        self.Ts = Ts
        self.Fys = Fys

    def unpack(self):
        """
        Returns variables in the order:
        D, W, f_r, w, endmill, Ts, Fys
        """
        return super(Data, self).unpack() + [self.Ts, self.Fys]

    def conditions(self):
        """
        Pulls conditions from a Data object.
        """
        return Conditions(*super().unpack())

    def __str__(self):
        info = {"D": "{:.5f}".format(self.D),
                "W": "{:.5f}".format(self.W),
                "f_r": "{:.5f}".format(self.f_r),
                "w": "{:.5f}".format(self.w),
                "endmill": str(self.endmill),
                "Ts average": "{:.5f}".format(np.median(self.Ts[:, 1])),
                "Fys average": "{:.5f}".format(np.median(self.Fys[:, 1]))}
        return "Data: (" + ", ".join([(name + ": " + value) for name, value in info.items()]) + ")"


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
        """
        Returns variables in the order:
        D, W, f_r, w, endmill, T, F
        """
        return super(Prediction, self).unpack() + [self.T, self.F]

    def conditions(self):
        """
        Pulls conditions from a Prediction object.
        """
        return Conditions(*super().unpack())

    def __str__(self):
        info = {"D": "{:.5f}".format(self.D),
                "W": "{:.5f}".format(self.W),
                "f_r": "{:.5f}".format(self.f_r),
                "w": "{:.5f}".format(self.w),
                "endmill": str(self.endmill),
                "T": "{:.5f}".format(self.T),
                "F": "{:.5f}, {:.5f}".format(*self.F)}

        return "Prediction: (" + ", ".join([(name + ": " + value) for name, value in info.items()]) + ")"
