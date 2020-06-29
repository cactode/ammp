"""
Intended to be the brain box of this project.
"""
import numpy as np
import time

from objects import EndMill, Conditions, MachineChar, Prediction
from fake_cut import ReplayCut
from ml import UnifiedLinearModel
from optimize import Optimizer

from models import deflection_load_simple, deflection_load

import logging
logging.basicConfig(level="INFO")

MACHINE_PORT = '/dev/ttyS25' 
SPINDLE_PORT = '/dev/ttyS33'
TFD_PORT     = '/dev/ttyS36'

D = 1e-3
W = 1e-3
f_r = 0.001
w = 100
D_A = 0.1e-3
N = 100
CONFIDENCE_RATE = np.linspace(0.2, 1, 10)

MACHINE = MachineChar(
    r_e = 1, 
    K_T = 0.10281,
    R_w = 0.188,
    V_max = 48,
    I_max = 10,
    T_nom = 0.12,
    f_r_max = 0.010,
    K_machine = 5e6,
    D_a = D_A
)

ENDMILL = EndMill(3, 3.175e-3, 3.175e-3, 19.05e-3, 1e-3)

FIXED_CONDITIONS = Conditions(
    D = D,
    W = W,
    f_r = f_r,
    w = w,
    endmill = ENDMILL
)
print(deflection_load_simple(D_A, Prediction(*FIXED_CONDITIONS.unpack(), 0.1, [100,100]), MACHINE))
print(deflection_load(D_A, Prediction(*FIXED_CONDITIONS.unpack(), 0.1, [100,100]), MACHINE))