"""
Intended to be the brain box of this project.
"""
FAKE = False

import numpy as np
import time
import shelve
import logging
import os

# hacky fix for issues with serial library
if not FAKE:
    from cut import Cut
from objects import EndMill, Conditions, MachineChar
from fake_cut import ReplayCut
from models import T_lin, F_lin, T_lin_full, F_lin_full
from ml import UnifiedLinearModel, UnifiedLinearModelFull
from optimize import Optimizer #, OptimizerFull, OptimizerPSO, OptimizerPSOFull

logging.basicConfig(level="INFO")

MACHINE_PORT = '/dev/ttyS25' 
SPINDLE_PORT = '/dev/ttyS33'
TFD_PORT     = '/dev/ttyS36'

# input variables
D = 1e-3 # depth of cut (always unchanging...)
W = 1e-3 # initial width of cut for bootstrap
f_r = 0.001 # initial feedrate for bootstrap
f_r_clearing = 0.003 # feedrate for facing and cutting start groove
w = 200 # spindle speed
START_DEPTH = 0.0e-3
START_FACE_D = 0.2e-3
ENDMILL = EndMill(3, 3.175e-3, 3.175e-3, 12e-3, 5e-3)
# ENDMILL = EndMill(3, 9.525e-3/2, 9.525e-3/2, 12.7e-3, 5e-3)
D_A = 0.1e-3 # maximum allowable deflection
N = 50 # total number of cuts to take, including bootstrap cuts
CONFIDENCE_RATE = np.linspace(0.2, 1, 5) # confidence progression during bootstrap cuts
USE_OLD_DATA = False

NAME = "ammp-lcs-1_4" # name to save to / name to draw data from if doing a fake cut
MODEL = UnifiedLinearModel
EQUATIONS = (T_lin, F_lin)
OPTIMIZER = Optimizer # optimizer to use

# taig machine
MACHINE = MachineChar(
    r_e = 1, 
    K_T = 0.10281,
    R_w = 0.188,
    V_max = 48,
    I_max = 10,
    T_nom = 0.12,
    f_r_max = 0.01,
    K_machine = 1.25e6,
    D_a = D_A
)

FIXED_CONDITIONS = Conditions(
    D = D,
    W = W,
    f_r = f_r,
    w = w,
    endmill = ENDMILL
)

logging.info("Initializing all structures")

cut = None
if FAKE:
    cut = ReplayCut(NAME, MODEL(), *EQUATIONS, [0,0], [0.1,2])
else:
    cut = Cut(MACHINE_PORT, SPINDLE_PORT, TFD_PORT, ENDMILL, 80e-3, 50.8e-3, f_r_clearing, w, START_DEPTH, NAME)
model = MODEL()
optimizer = OPTIMIZER(model, MACHINE, D_A, FIXED_CONDITIONS)

logging.info("Beginning facing operation, creating starting groove")
if START_FACE_D:
    cut.face_layer(START_FACE_D)
cut.begin_layer(D)

logging.info("First bootstrap cut to obtain a basic characterization")

conditions_conservative = Conditions(D, W, f_r, w, ENDMILL)
datum = cut.cut(conditions_conservative, save=True, auto_layer=True)
model.ingest_datum(datum)

logging.info("After bootstrap cut, model params are actually at: " + ", ".join(["{:.5e}".format(p) for p in model.params]))
if USE_OLD_DATA and not FAKE:
    with shelve.open(os.path.join("saved_cuts", "db")) as db:
        model.ingest_data(db[USE_OLD_DATA])
logging.info("Partially optimized bootstrap cuts starting now")

# start optimizing, but only slowly start accepting new datums
confidences = list(CONFIDENCE_RATE) + [1] * (N - len(CONFIDENCE_RATE))
for confidence in confidences:
    logging.info("Confidence at    : " + str(confidence))
    conditions_optimized = optimizer.optimize(verbose = True)
    logging.info("Optimized        : " + str(conditions_optimized))
    conditions_compromise = conditions_conservative.compromise(conditions_optimized, confidence)
    logging.info("Compromised      : " + str(conditions_compromise))
    logging.info("Model guesses    : " + str(model.predict_one(conditions_compromise)))
    datum = cut.cut(conditions_compromise, save = True, auto_layer=True)
    logging.info("Datum obtained   : " + str(datum))
    model.ingest_datum(datum)
    logging.info("Params updated to: " + ", ".join(["{:.5e}".format(p) for p in model.params]))

if FAKE:
    logging.info("Actual cut params: " + ", ".join(["{:.5e}".format(p) for p in cut.params]))