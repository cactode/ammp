from sensors import TFD, Spindle_Applied

import time
import logging
logging.basicConfig(level=logging.DEBUG)

TFD_PORT = '/dev/ttyS26'
SPN_port = '/dev/ttyS33'

TEST_CALIBRATION = 1

def test_tfd():
    tfd = TFD(TFD_PORT)
    max_force = 0
    while True:
        force = tfd.get_force() / TEST_CALIBRATION
        if force > max_force: max_force = force
        print("Force from TFD:", force)
        print("Max force: ", max_force)


def test_spindle():
    spindle = Spindle_Applied(SPN_port)
    spindle.set_w(300)
    # spindle.calibrate()
    while True: pass
    print("Calibrated to ", spindle.torque_loss)
    spindle.set_w(0)

test_spindle()
