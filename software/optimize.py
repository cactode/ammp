import numpy as np

from scipy.optimize import minimize
from functools import reduce

from models import (
    optimality,
    deflection_load,
    failure_prob_milling,
    motor_torque_load,
    motor_speed_load
)

from objects import MachineChar, Prediction, Conditions
from ml import Model


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

    MAX_TOOL_BREAKAGE_PROB = 0.10

    def __init__(self, model: Model, machinechar: MachineChar, D_a, fixed_conditions: Conditions):

        self.model = model
        self.machinechar = machinechar
        self.D_a = D_a
        self.fixed_conditions = fixed_conditions

    def optimize(self):
        """
        Uses the current state of the model to predict the optimum milling conditions.
        """
        x0 = [self.fixed_conditions.W, self.fixed_conditions.f_r]
        bounds = ((0, self.fixed_conditions.endmill.r_c * 2 - 1e-10),
                  (0, self.machinechar.f_r_max))
        minimum = minimize(self._optimize_func, x0, bounds=bounds)
        W, f_r = minimum.x
        D, _, _, w, endmill = self.fixed_conditions.unpack()
        return Conditions(D, W, f_r, w, endmill)

    def _optimize_func(self, X):
        W, f_r = X
        D, _, _, w, endmill = self.fixed_conditions.unpack()
        conditions = Conditions(D, W, f_r, w, endmill)
        prediction = self.model.predict_one(conditions)
        return self.loss(prediction)

    def loss(self, prediction: Prediction):
        """
        Find the "loss" for this problem.
        Loss is defined as the inverse of (optimality * failure metric).
        """
        return - (optimality(prediction.conditions()) * self.failure(prediction))

    def inv_logistic(self, x):
        return 1 - (1 / (1 + np.exp(-20 * (x - 0.9))))

    def failure(self, prediction: Prediction):
        """
        Failure metric. Slightly arbitrary. 
        Failure is defined as the geometric combination of all failure modes 
            (each normalized between 0-1 using the logistic function.)
        """
        # pull out all of these
        failures = [deflection_load(self.D_a, prediction),
                    # since we want some failure margin
                    failure_prob_milling(
                        prediction) + (1 - Optimizer.MAX_TOOL_BREAKAGE_PROB) - 0.10,
                    motor_torque_load(prediction, self.machinechar),
                    motor_speed_load(prediction, self.machinechar)
                    ]

        failure_logistic = reduce(
            lambda x, y: x*y, map(self.inv_logistic, failures))
        return failure_logistic
