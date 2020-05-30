import numpy as np
import time
import shelve
import os

from queue import PriorityQueue
from objects import Data, EndMill
from sensors import Machine, Spindle_Applied, TFD

import logging
log = logging.getLogger(__name__)


class MachineCrash(Exception):
    pass


class Cut:
    def __init__(self, machine_port, spindle_port, tfd_port, endmill, x_max, y_max, initial_z=0, save_as=None):
        self.machine = Machine(machine_port)
        self.spindle = Spindle_Applied(spindle_port)
        self.tfd = TFD(tfd_port)
        log.info("Initialized all systems.")

        # saving other variables
        self.endmill = endmill
        self.x_max = x_max
        self.y_max = y_max
        self.cut_x = 0
        self.cut_z = initial_z
        self.D = 0
        self.save_as = save_as

        # deriving some constants
        self.Y_START = - 2 * endmill.R
        self.Y_END = y_max + 2 * endmill.R
        self.X_END = x_max - endmill.R
        self.Y_DATA_START = 1.5 * endmill.R
        self.Y_DATA_END = y_max - 1.5 * endmill.R

    def __del__(self):
        self.spindle.set_w(0)

    def begin_layer(self, D, f_r_clearing, w_clearing):
        """
        Prepares to start facing with cut depth D. Successive calls will compensate for previous cut depths.
        Args:
            D: Depth of cut for this layer.
            f_r_clearing: feedrate used for this clearing pass.
            w_clearing: spindle speed used for this clearing pass.

        Returns:
            A data blob from this operation.
        """
        X_START = self.endmill.R
        log.info("Preparing to clear layer to depth " + str(D) +
                 " at feedrate " + str(f_r_clearing) + " with speed " + str(w_clearing))
        self.D = D
        self.cut_z -= D

        self.spindle.set_w(w_clearing)

        self.machine.rapid({'x': X_START, 'y': self.Y_START})
        self.machine.rapid({'z': self.cut_z})
        self.machine.cut({'y': self.Y_END}, f_r_clearing)
        self.machine.rapid({'z': self.cut_z + D + 1})
        self.machine.rapid({'x': X_START, 'y': self.Y_START})
        self.machine.rapid({'z': self.cut_z})
        self.machine.hold_until_still()

        self.x_cut = self.endmill.r_c * 2

        log.info("Layer prepared for clearing")

    def cut(self, conditions):
        """
        Performs a stroke of facing. Returns a data blob.
        """
        _, W, f_r, w, _ = conditions.unpack()
        X_START = self.x_cut - self.endmill.r_c + W
        if X_START > self.X_END:
            raise MachineCrash(
                "Cutting too far in X direction: X = " + str(X_START))

        outQueue = PriorityQueue()

        log.info("Performing cut at position " + str(X_START) + " with WOC " +
                 str(W) + " and feedrate " + str(f_r) + " at speed " + str(w))

        self.machine.rapid({'x': X_START, 'y': self.Y_START})
        self.machine.rapid({'z': self.cut_z})
        self.machine.hold_until_still()

        log.info("Calibrating spindle")
        self.spindle.calibrate()
        log.info("Calibrating TFD")
        self.tfd.calibrate()

        log.info("Starting cut")
        self.machine.cut({'y': self.Y_END}, f_r)
        time.sleep(0.1)
        self.machine.hold_until_condition(
            lambda state: state['wpos']['y'] > self.Y_DATA_START)

        log.info("Beginning data collection")
        time_start = time.perf_counter
        self.spindle.start_measurement(outQueue, time_start)
        self.tfd.start_measurement(outQueue, time_start)
        self.machine.hold_until_condition(
            lambda state: state['wpos']['y'] > self.Y_DATA_END)

        log.info("Ending data collection")
        self.spindle.stop_measurement()
        self.tfd.stop_measurement()

        log.info("Finished cut, collected " +
                 str(len(outQueue)) + " data points.")

        self.machine.rapid({'z': self.cut_z + self.D + 1})
        self.machine.rapid({'y': self.Y_START})

        self.x_cut += W

        Ts, Fys = self.dump_queue(outQueue)

        data = Data(self.D, W, f_r, w, self.endmill, Ts, Fys)
        if self.save_as:
            with shelve.open(os.path.join("saved_cuts", "db")) as db:
                if self.save_as in db:
                    db[self.save_as].append(data)
                else:
                    db[self.save_as] = [data]

    def dump_queue(self, outQueue):
        Ts, Fys = list(), list()
        while not outQueue.empty():
            t, datum = outQueue.get()
            datum_type, datum_value = datum
            if datum_type == 'spindle':
                Ts.append(t, datum_value)
            elif datum_type == 'tfd':
                Fys.append(t, datum_value)

        return np.array(Ts), np.array(Fys)


if __name__ == "__main__":
    endmill = EndMill(3, 3.175e-3, 3.175e-3, 0.019, 0.004)
    cut = Cut('/dev/ttyS28', '/dev/ttyS26', 'dev/ttyS27', endmill, 50e-3, 50e-3)
    
