import numpy as np
import pyswarms

from models import optimality


def loss(self, prediction):
    D = prediction.D
    W = prediction.W
    opt = optimality(D, W, f_r)

