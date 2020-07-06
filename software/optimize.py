import numpy as np

from scipy.optimize import minimize
from functools import reduce

from models import (
    optimality,
    deflection_load,
    deflection_load_simple,
    failure_prob_milling,
    motor_torque_load,
    motor_speed_load
)

from objects import MachineChar, Prediction, Conditions
from ml import Model
import pyswarms as ps

import logging
log = logging.getLogger(__name__)


class Optimizer:
    """
    Initializes optimizer with constraints.

    Args:
        model : A Model representing the current milling process.
        machinechar : A MachineChar representing the machine being used.
        D_a: Maximum allowable deflection
        fixed_conditions: A conditions object that defines the values for things that cannot be changed. 
                          W and f_r are only considered as initial guesses for the optimizer.
    """

    MAX_TOOL_BREAKAGE_PROB = 0.05

    def __init__(self, model: Model, machinechar: MachineChar, D_a, fixed_conditions: Conditions):

        self.model = model
        self.machinechar = machinechar
        self.D_a = D_a
        self.fixed_conditions = fixed_conditions

    def optimize(self, verbose=False):
        """
        Uses the current state of the model to predict the optimum milling conditions.
        """
        x0 = [self.fixed_conditions.W, self.fixed_conditions.f_r]
        bounds = ((0, self.fixed_conditions.endmill.r_c * 1.8),
                  (0, self.machinechar.f_r_max))
        minimum = minimize(self._optimize_func, x0, bounds=bounds)
        W, f_r = minimum.x
        D, _, _, w, endmill = self.fixed_conditions.unpack()
        optimized = Conditions(D, W, f_r, w, endmill)
        if verbose:
            prediction = self.model.predict_one(optimized)
            self.failure(prediction, verbose)
        return optimized

    def _optimize_func(self, X):
        W, f_r = X
        D, _, _, w, endmill = self.fixed_conditions.unpack()
        conditions = Conditions(D, W, f_r, w, endmill)
        prediction = self.model.predict_one(conditions)
        loss = self.loss(prediction)
        return loss

    def loss(self, prediction: Prediction):
        """
        Find the "loss" for this problem.
        Loss is defined as the inverse of (optimality * failure metric).
        This is boosted by 1e8 to deal with floating point precision.
        """
        return - (1e8 * (optimality(prediction.conditions())) * self.failure(prediction))

    def inv_logistic(self, x):
        """
        Inverse logistic function that smoothly goes from 1 -> 0. Centered at 0.9.
        """
        return 1 - (1 / (1 + np.exp(-20 * (x - 0.9))))

    def failure(self, prediction: Prediction, verbose=False):
        """
        Failure metric. Slightly arbitrary. At 1 when failure is improbable and at 0 when failure is likely.
        Failure is defined as the geometric combination of all failure modes 
            (each normalized between 0-1 using the logistic function.)
        """
        # pull out all of these
        failure_deflection = deflection_load_simple(
            prediction, self.machinechar)
        failure_breakage = failure_prob_milling(
            prediction) / Optimizer.MAX_TOOL_BREAKAGE_PROB
        failure_torque = motor_torque_load(prediction, self.machinechar)
        failure_speed = motor_speed_load(prediction, self.machinechar)

        failures = [failure_deflection,
                    failure_breakage,
                    failure_torque,
                    failure_speed]

        if verbose:
            log.info("Ind. failures: " + ", ".join([name + ": " + "{:.3f}".format(
                value) for name, value in zip(['deflection', 'breakage', 'torque', 'speed'], failures)]))

        failure_logistic = reduce(
            lambda x, y: x*y, map(self.inv_logistic, failures))
        return failure_logistic


class OptimizerFull(Optimizer):
    """Optimizer that uses cutting models that try to account for variations in cutting pressures due to different cutting speeds. Doesn't really work...
    """

    def optimize(self, verbose=False):
        """
        Uses the current state of the model to predict the optimum milling conditions.
        """
        x0 = [self.fixed_conditions.W,
              self.fixed_conditions.f_r, self.fixed_conditions.w]
        bounds = ((0, self.fixed_conditions.endmill.r_c * 1.8),
                  (0, self.machinechar.f_r_max),
                  (0, None))
        minimum = minimize(self._optimize_func, x0, bounds=bounds)
        W, f_r, w = minimum.x
        D, _, _, _, endmill = self.fixed_conditions.unpack()
        optimized = Conditions(D, W, f_r, w, endmill)
        if verbose:
            prediction = self.model.predict_one(optimized)
            self.failure(prediction, verbose=True)
        return optimized

    def _optimize_func(self, X):
        W, f_r, w = X
        D, _, _, _, endmill = self.fixed_conditions.unpack()
        conditions = Conditions(D, W, f_r, w, endmill)
        prediction = self.model.predict_one(conditions)
        loss = self.loss(prediction)
        return loss


class OptimizerPSO(Optimizer):
    """Uses a particle swarm optimizer. Doesn't really work.
    """

    def optimize(self, verbose=False):
        """
        Uses the current state of the model to predict the optimum milling conditions.
        """
        min_bound = np.array([0, 0])
        max_bound = np.array([self.fixed_conditions.endmill.r_c * 1.8,
                              self.machinechar.f_r_max])
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.global_best.GlobalBestPSO(
            100, 2, options=options, bounds=(min_bound, max_bound))

        def func(x): return np.apply_along_axis(self._optimize_func, 1, x)
        _, pos = optimizer.optimize(func, iters=500)
        W, f_r = pos
        D, _, _, w, endmill = self.fixed_conditions.unpack()
        optimized = Conditions(D, W, f_r, w, endmill)
        if verbose:
            prediction = self.model.predict_one(optimized)
            self.failure(prediction, verbose=True)
        return optimized

    def _optimize_func(self, X):
        W, f_r = X
        D, _, _, w, endmill = self.fixed_conditions.unpack()
        conditions = Conditions(D, W, f_r, w, endmill)
        prediction = self.model.predict_one(conditions)
        loss = self.loss(prediction)
        return loss
