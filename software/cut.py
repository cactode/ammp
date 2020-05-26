import numpy as np
import logging
import time

from queue import PriorityQueue
from objects import Data, EndMill
from sensors import Machine, Applied_Spindle, TFD


class MachineCrash(Exception):
    pass


class Cut:
    def __init__(self, machine_port, spindle_port, tfd_port, endmill, x_max, y_max, initial_z = 0):
        self._logger = logging.Logger("Cut")
        self.machine = Machine(machine_port)
        self.spindle = Applied_Spindle(spindle_port)
        self.tfd = TFD(tfd_port)
        self._logger.info("Initialized all systems.")

        # saving other variables
        self.endmill = endmill
        self.x_max = x_max
        self.y_max = y_max
        self.cut_x = 0
        self.cut_z = initial_z
        self.D = 0

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
        """
        X_START = self.endmill.R
        self._logger.info("Preparing to clear layer to depth " + str(D) + " at feedrate " + str(f_r_clearing) + " with speed " + str(w_clearing))
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

        self._logger.info("Layer prepared for clearing")

    
    def cut(self, conditions):
        """
        Performs a stroke of facing. Returns a data blob.
        """
        _, W, f_r, w, _ = conditions.unpack()
        X_START = x_cut - self.endmill.r_c + W
        if X_START > self.X_END:
            raise MachineCrash("Cutting too far in X direction: X = " + str(X_START))

        outQueue = PriorityQueue()

        self._logger.info("Performing cut at position " + str(X_START) + " with WOC " + str(W) + " and feedrate " + str(f_r) + " at speed " + str(w))

        self.machine.rapid({'x': X_START, 'y': self.Y_START})
        self.machine.rapid({'z': self.cut_z})
        self.machine.hold_until_still()

        self._logger.info("Calibrating spindle")
        self.spindle.calibrate()
        self._logger.info("Calibrating TFD")
        self.tfd.calibrate()
        
        self._logger.info("Starting cut")
        self.machine.cut({'y': self.Y_END}, f_r)
        time.sleep(0.1)
        self.machine.hold_until_condition(lambda state : state['wpos']['y'] > self.Y_DATA_START)

        time_start = time.perf_counter
        self.spindle.start_measurements(outQueue, time_start)
        self.tfd.start_measurement(outQueue, time_start)
        self.machine.hold_until_condition(lambda state : state['wpos']['y'] > self.Y_DATA_END)

        self.spindle.stop_measurements()
        self.tfd.stop_measurement()

        self._logger.info("Finished cut, collected " + str(len(outQueue)) + " data points.")

        self.machine.rapid({'z': self.cut_z + self.D + 1})
        self.machine.rapid({'y': self.Y_START})

        self.x_cut += W

        Ts, Fys = self.dump_queue(outQueue)

        return Data(self.D, W, f_r, w, self.endmill, Ts, Fys)

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
    logging.basicConfig(level=logging.INFO)