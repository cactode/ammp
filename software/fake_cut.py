from models import T_lin, F_lin
from objects import Conditions, Data
import numpy as np
import shelve
import os


class Fake_Cut:
    """
    Fake cutting process. Returns results using prebaked parameters and specified noise levels.

    Args:
        params: list of format [K_tc, K_te, K_rc, K_re]
        error: list of standard deviations of format [o_T, o_Fy]. Simulates the sensor being "wrong" sometimes.
        error: list of standard deviations of format [o_T, o_Fy]. Simulates the sensor being noisy.
    """

    def __init__(self, params, error, noise):
        self.params = params
        self.error = error
        self.noise = noise

    def begin_cut(self, *args, **kwargs):
        pass

    def cut(self, conditions: Conditions):
        # use prediction as output
        T = T_lin(conditions, *self.params[:2])
        _, Fy = F_lin(conditions, *self.params)
        # add sensor error
        T_error = np.random.normal(T, self.error[0])
        Fy_error = np.random.normal(Fy, self.error[1])
        # add sensor noise
        T_noisy = np.random.normal(T_error, self.noise[0], (100))
        Fy_noisy = np.random.normal(Fy_error, self.noise[1], (100))
        # generate fake times
        t = np.linspace(0, 1, 100)
        # return fake reading
        return Data(*conditions.unpack(), np.array([t, T_noisy]), np.array([t, Fy_noisy]))


class ReplayCut:
    def __init__(self, replay_data):
        self.data = None
        with shelve.open(os.path.join("saved_data", "db")) as db:
            self.data = db[replay_data]

    def begin_cut(self, *args, **kwargs):
        pass

    def cut(self, *args, **kwargs):
        for datum in self.data:
            yield datum
