from models import T_lin, F_lin, T_lin_full, F_lin_full
from objects import Conditions, Data
import numpy as np
import shelve
import os
from ml import UnifiedLinearModel

import logging
log = logging.getLogger(__name__)

class Fake_Cut:
    """
    Fake cutting process. Returns results using prebaked parameters and specified noise levels.

    Args:
        params: list of params
        T_func: A function to use for torques
        F_func: A function to use for forces
        error: list of standard deviations of format [o_T, o_Fy]. Simulates the sensor being "wrong" sometimes.
        error: list of standard deviations of format [o_T, o_Fy]. Simulates the sensor being noisy.
    """

    def __init__(self, params, T_func, F_func, error, noise):
        self.params = params
        self.T_func = T_func
        self.F_func = F_func
        self.error = error
        self.noise = noise
        log.info("Initialized fake cut with params: " + str(self.params))

    def face_layer(self, *args, **kwargs):
        log.info("Face cut called")
        pass

    def begin_layer(self, *args, **kwargs):
        log.info("Begin cut called")
        pass

    def cut(self, conditions: Conditions, *args, **kwargs):
        # use prediction as output
        T = self.T_func(conditions, *self.params)
        _, Fy = self.F_func(conditions, *self.params)
        # add sensor error
        T_error = np.random.normal(T, self.error[0])
        Fy_error = np.random.normal(Fy, self.error[1])
        # add sensor noise
        T_noisy = np.random.normal(T_error, self.noise[0], (100))
        Fy_noisy = np.random.normal(Fy_error, self.noise[1], (100))
        # generate fake times
        t = np.linspace(0, 1, 100)
        # return fake reading
        data = Data(*conditions.unpack(), np.array([t, T_noisy]).T, np.array([t, Fy_noisy]).T)
        return data

    def scale_coefs(self, scale):
        self.params = [scale * p for p in self.params]


class ReplayCut(Fake_Cut):
    def __init__(self, replay_data, model, T_func, F_func, error, noise):
        self.model = model
        with shelve.open(os.path.join("saved_cuts", "db")) as db:
            data = db[replay_data]
            self.model.ingest_data(data)
        super().__init__(self.model.params, T_func, F_func, error, noise)