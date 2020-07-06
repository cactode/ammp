"""
A tiny utility script that just collects sweep data.
"""
import numpy as np
import time

from objects import EndMill, Conditions, MachineChar
from cut import Cut
from optimize import Optimizer

import logging

logging.basicConfig(level="INFO")

MACHINE_PORT = "/dev/ttyS25"
SPINDLE_PORT = "/dev/ttyS33"
TFD_PORT = "/dev/ttyS36"

endmill = EndMill(3, 3.175e-3, 3.175e-3, 12e-3, 5e-3)

fixed_conditions = Conditions(
    D=1e-3, W=1e-3, f_r=0.001, w=300, endmill=endmill)

cut = Cut(
    MACHINE_PORT,
    SPINDLE_PORT,
    TFD_PORT,
    endmill,
    60e-3,
    50.8e-3,
    5e-3,
    300,
    initial_z=0.5e-3,
    save_as="sweep-alu-1_4-v2",
)

f_r_range = np.linspace(2e-3, 0.01, 6)
W_range = np.linspace(1e-3, 3.175e-3 * 1.8, 6)

cut.face_layer(D=0.3e-3)

cut.begin_layer(D=1e-3)

for f_r in f_r_range:
    for W in W_range:
        conditions = Conditions(1e-3, W, f_r, 200, endmill)
        cut.cut(conditions, save=True, auto_layer=True)

cut.close()
