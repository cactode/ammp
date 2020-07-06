"""
Intended to be the brain box of this project.
"""
from optimize import Optimizer
from ml import UnifiedLinearModel
from models import T_lin, F_lin
from objects import EndMill, Conditions, MachineChar
import os
import logging
import shelve
import time
import numpy as np
FAKE = True


# hacky fix for issues with serial library
if FAKE:
    from fake_cut import ReplayCut
else:
    from cut import Cut


logging.basicConfig(level="INFO")


def ammp(MACHINE_PORT, SPINDLE_PORT, TFD_PORT, D, W, f_r, f_r_clearing, w, START_DEPTH, START_FACE_D, ENDMILL, D_A, N, X_TRAVEL, CONFIDENCE_RATE, USE_OLD_DATA, NAME, MODEL, EQUATIONS, OPTIMIZER, MACHINE):
    """
    Runs the optimizing system. 
    System starts out by facing the stock flat (since we need to make sure the endmill bottom perfectly corresponds to the start of the stock).
    Then does a bootstrap cut to initialize the model.
    Then starts optimizing process itself.

    Args:
        MACHINE_PORT: port name for machine
        SPINDLE_PORT: port name for spindle
        TFD_PORT: port name for tool force dyno
        D: depth of cut (always unchanging...)
        W: initial width of cut for bootstrap
        f_r: initial feedrate for bootstrap
        f_r_clearing: feedrate for facing and cutting start groove
        w: spindle speed
        START_DEPTH: offset depth to start cutting at
        START_FACE_D: how deep to face stock before starting ammp runs
        ENDMILL: endmill parameters
        D_A: maximum allowable deflection
        N: total number of cuts to take, including bootstrap cuts
        X_TRAVEL: travel in the x direction
        CONFIDENCE_RATE: confidence progression during bootstrap cuts
        USE_OLD_DATA: whether or not to use datapoints from an older run. change this to run name to use this feature
        NAME: name to save to / name to draw data from if doing a fake cut
        MODEL: model class to use
        EQUATIONS: equations to use in optimizer
        OPTIMIZER: optimizer to use
        MACHINE: machine characteristics
    """
    FIXED_CONDITIONS = Conditions(
        D=D,
        W=W,
        f_r=f_r,
        w=w,
        endmill=ENDMILL
    )

    logging.info("Initializing all structures")

    cut = None
    if FAKE:
        cut = ReplayCut(NAME, MODEL(), *EQUATIONS, [0, 0], [0.1, 2])
    else:
        cut = Cut(MACHINE_PORT, SPINDLE_PORT, TFD_PORT, ENDMILL, X_TRAVEL,
                  50.8e-3, f_r_clearing, w, START_DEPTH, NAME, graceful_shutdown=True)
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

    logging.info("After bootstrap cut, model params are actually at: " +
                 ", ".join(["{:.5e}".format(p) for p in model.params]))
    if USE_OLD_DATA and not FAKE:
        with shelve.open(os.path.join("saved_cuts", "db")) as db:
            model.ingest_data(db[USE_OLD_DATA])
    logging.info("Partially optimized bootstrap cuts starting now")

    # start optimizing, but only slowly start accepting new datums
    confidences = list(CONFIDENCE_RATE) + [1] * (N - len(CONFIDENCE_RATE))
    for i, confidence in enumerate(confidences):
        logging.info("------------------ Run #" +
                     str(i+1) + " -----------------------")
        logging.info("Confidence at    : " + str(confidence))
        conditions_optimized = optimizer.optimize(verbose=True)
        logging.info("Optimized        : " + str(conditions_optimized))
        conditions_compromise = conditions_conservative.compromise(
            conditions_optimized, confidence)
        logging.info("Compromised      : " + str(conditions_compromise))
        logging.info("Model guesses    : " +
                     str(model.predict_one(conditions_compromise)))
        datum = cut.cut(conditions_compromise, save=True, auto_layer=True)
        logging.info("Datum obtained   : " + str(datum))
        model.ingest_datum(datum)
        logging.info("Params updated to: " +
                     ", ".join(["{:.5e}".format(p) for p in model.params]))

    if FAKE:
        logging.info("Actual cut params: " +
                     ", ".join(["{:.5e}".format(p) for p in cut.params]))


if __name__ == "__main__":
    # input variables
    MACHINE_PORT = '/dev/ttyS25'
    SPINDLE_PORT = '/dev/ttyS33'
    TFD_PORT = '/dev/ttyS36'
    D = 1.5e-3
    W = 1e-3
    f_r = 0.001
    f_r_clearing = 0.005
    w = 200
    START_DEPTH = 0e-3
    START_FACE_D = 0.1e-3
    ENDMILL = EndMill(3, 3.175e-3 / 2, 3.175e-3 / 2, 10e-3, 5e-3)
    # ENDMILL = EndMill(3, 3.175e-3, 3.175e-3, 12e-3, 5e-3)
    # ENDMILL = EndMill(3, 9.525e-3/2, 9.525e-3/2, 20e-3, 5e-3)
    D_A = 0.1e-3
    N = 10
    X_TRAVEL = 30e-3
    CONFIDENCE_RATE = np.linspace(0.2, 1, 5)
    USE_OLD_DATA = False

    NAME = "ammp-alu-1_8_coolant"
    MODEL = UnifiedLinearModel
    EQUATIONS = (T_lin, F_lin)
    OPTIMIZER = Optimizer

    # taig machine
    MACHINE = MachineChar(
        r_e=1,
        K_T=0.10281,
        R_w=0.188,
        V_max=48,
        I_max=10,
        T_nom=0.12,
        f_r_max=0.01,
        K_machine=1.25e6,
        D_a=D_A
    )
    ammp(MACHINE_PORT, SPINDLE_PORT, TFD_PORT, D, W, f_r, f_r_clearing, w, START_DEPTH, START_FACE_D,
         ENDMILL, D_A, N, X_TRAVEL, CONFIDENCE_RATE, USE_OLD_DATA, NAME, MODEL, EQUATIONS, OPTIMIZER, MACHINE)
