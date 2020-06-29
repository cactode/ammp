"""
Intended to be the brain box of this project; for now, it just runs a test sweep to collect initial data.
"""
import numpy as np
import time

from objects import EndMill, Conditions, MachineChar
from cut import Cut
from ml import LinearModel
from optimize import Optimizer

import logging
logging.basicConfig(level="INFO")

MACHINE_PORT = '/dev/ttyS25' 
SPINDLE_PORT = '/dev/ttyS33'
TFD_PORT     = '/dev/ttyS36'
IGNORE_FIRST = 0

machine = MachineChar(
    r_e = 1, 
    K_T = 0.10281,
    R_w = 0.188,
    V_max = 48,
    I_max = 10,
    T_nom = 0.12,
    f_r_max = 0.017,
    K_machine = 5e6,
    D_a = 1
)
endmill = EndMill(3, 3.175e-3, 3.175e-3, 9.5e-3, 3e-3)

fixed_conditions = Conditions(
    D = 1e-3,
    W = 1e-3,
    f_r = 0.001,
    w = 300,
    endmill = endmill
)

cut = Cut(MACHINE_PORT, SPINDLE_PORT, TFD_PORT, endmill, 80e-3, 50.8e-3, 5e-3, 300, save_as = "6061-sweep-speed")

f_r_range = np.linspace(2e-3, 0.011, 4)
W_range = np.linspace(1e-3, 3.175e-3 * 1.8, 5)
w_range = np.linspace(100, 300, 5)


cut.face_layer(D = 0.3e-3)

cut.begin_layer(D = 1e-3)

for f_r in f_r_range: 
    for W in W_range:
        for w in w_range:
            conditions = Conditions(1e-3, W, f_r, w, endmill)
            cut.cut(conditions, save = True, auto_layer=True)
        
cut.close()