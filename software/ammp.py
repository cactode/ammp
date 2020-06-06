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
TFD_PORT     = '/dev/ttyS35'

endmill = EndMill(3, 3.175e-3, 3.175e-3, 19.05e-3, 1e-3)

cut = Cut(MACHINE_PORT, SPINDLE_PORT, TFD_PORT, endmill, 80e-3, 50.8e-3, 5e-3, 300, save_as = "6061-sweep")

f_r_range = np.linspace(2e-3, 5e-3, 9)
W_range = np.linspace(1e-3, 3.175e-3, 10)

cut.face_layer(D = 0.3e-3)

cut.begin_layer(D = 1e-3)

for f_r in f_r_range: 
    for W in W_range:
        conditions = Conditions(1e-3, W, f_r, 300, endmill)
        cut.cut(conditions, save = True, auto_layer=True)
        
cut.close()