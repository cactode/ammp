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

def inv_logistic(x, center, scale):
    return 1 - (1 / 1 + np.exp(-scale * (x - center)))

class Optimizer:
    def __init__(self, model : Model, machinechar : MachineChar, D_a, fixed_conditions : Conditions):
        """
        Initializes optimizer with constraints.
        Args:
            model : A Model representing the current milling process.
            spindlechar : A SpindleChar representing the machine being used.
        """
        self.model = model
        self.spindlechar = spindlechar
        self.D_a = D_a
        self.fixed_conditions = fixed_conditions

    def optimize(self):
        x0 = [self.fixed_conditions.W, self.fixed_conditions.f_r]
        bounds = ((0, self.fixed_conditions.endmill.R * 2), (0, self.machinechar.f_r_max))
        minimum = minimize(self.optimize_func, x0, bounds = bounds)
        W, f_r = minimum.x
        D, _, _, w, endmill = self.fixed_conditions.unpack()
        return Conditions(D, W, f_r, w, endmill)

    def optimize_func(self, X):
        W, f_r = X
        D, _, _, w, endmill = self.fixed_conditions.unpack()
        conditions = Conditions(D, W, f_r, w, endmill)
        prediction = model.predict_one(conditions)
        return loss(prediction)

    def loss(self, prediction : Prediction):
        """
        Find the "loss" for this problem.
        Loss is defined as the inverse of (optimality * failure metric).
        """
        return - (optimality(prediction) * self.failure(prediction))

    def inv_logistic(self, x):
        return 1 - (1 / 1 + np.exp(-30 * (x - 0.9)))
       
    def failure(self, prediction : Prediction):
        """
        Failure metric. Slightly arbitrary.
        """
        # pull out all of these
        failures = [deflection_load(self.D_a, prediction), 
                    failure_prob_milling(prediction) + 0.80, # lol oops 
                    motor_torque_load(prediction, self.spindlechar),
                    motor_speed_load(prediction, self.spindlechar)
                   ]
        
        failure_logistic = reduce(lambda x,y:x*y, map(self.inv_logistic, failures))
        return failure



        