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
    def __init__(self, machine_port, spindle_port, tfd_port, endmill, x_max, y_max, f_r_clearing, w_clearing, initial_z=0, save_as=None):
        self.machine = Machine(machine_port)
        self.machine.unlock()
        self.machine.zero()
        self.spindle = Spindle_Applied(spindle_port)
        self.tfd = TFD(tfd_port)
        log.info("Initialized all systems.")

        # saving other variables
        self.endmill = endmill
        self.x_max = x_max
        self.y_max = y_max
        self.cut_x = 0
        self.cut_z = -initial_z # initial z in positive units
        self.D = 0
        self.f_r_clearing = f_r_clearing
        self.w_clearing = w_clearing
        self.save_as = save_as

        # deriving some constants
        self.Y_START = - 2 * endmill.r_c
        self.Y_END = y_max + 2 * endmill.r_c
        self.X_START = endmill.r_c
        self.X_END = x_max - endmill.r_c
        self.Y_DATA_START = 1.5 * endmill.r_c
        self.Y_DATA_END = y_max - 1.5 * endmill.r_c

    def __del__(self):
        self.spindle.set_w(0)

    def warmup_spindle(self):
        log.info("Warming up spindle")
        self.spindle.set_w(300)
        time.sleep(30)
        self.spindle.set_w(0)
        log.info("Finished warming spindle")

    def face_layer(self, D):
        """
        Faces a layer to ensure that all cuts afterwards are even.
        Args:
            D: Depth of cut for this layer.
            f_r_clearing: feedrate used for this clearing pass.
            w_clearing: spindle speed used for this clearing pass.

        """
        log.info("Preparing to face layer to depth " + str(D) +
                 " at feedrate " + str(self.f_r_clearing) + " with speed " + str(self.w_clearing))       

        # define next cut
        self.D = D
        self.cut_z -= D
        cuts = np.append(np.arange(self.X_START, self.X_END, 1.8 * self.endmill.r_c), self.X_END)
        
        # perform cut
        self.spindle.set_w(self.w_clearing)
        self.machine.rapid({'x': self.X_START, 'y': self.Y_START})
        self.machine.rapid({'z': self.cut_z})

        location = 'BOTTOM'

        for x in cuts:
            self.machine.rapid({'x': x})
            if location == 'BOTTOM':
                self.machine.cut({'y': self.Y_END}, self.f_r_clearing)
                location = 'TOP'
            elif location == 'TOP':
                self.machine.cut({'y': self.Y_START}, self.f_r_clearing)
                location = 'BOTTOM'
        self.machine.rapid({'z': self.cut_z + D + 1e-3})
        self.machine.rapid({'x': self.X_START, 'y': self.Y_START})
        self.machine.rapid({'z': self.cut_z})
        self.machine.hold_until_still()

        log.info("Workpiece faced")

    def begin_layer(self, D):
        """
        Prepares to start facing with cut depth D. Successive calls will compensate for previous cut depths.
        Args:
            D: Depth of cut for this layer (in positive units)
            f_r_clearing: feedrate used for this clearing pass.
            w_clearing: spindle speed used for this clearing pass.

        Returns:
            A data blob from this operation.
        """
        log.info("Preparing to clear layer to depth " + str(D) +
                 " at feedrate " + str(self.f_r_clearing) + " with speed " + str(self.w_clearing))
        self.D = D
        self.cut_z -= D

        self.spindle.set_w(self.w_clearing)

        self.machine.rapid({'x': self.X_START, 'y': self.Y_START})
        self.machine.rapid({'z': self.cut_z})
        self.machine.cut({'y': self.Y_END}, self.f_r_clearing)
        self.machine.rapid({'z': self.cut_z + D + 1e-3})
        self.machine.rapid({'x': self.X_START, 'y': self.Y_START})
        self.machine.rapid({'z': self.cut_z})
        self.machine.hold_until_still()

        self.x_cut = self.endmill.r_c * 2

        log.info("Layer prepared for clearing")

    def cut(self, conditions, save = True, auto_layer = True):
        """
        Performs a stroke of facing. Returns a data blob.
        """


        _, W, f_r, w, _ = conditions.unpack()

        self.spindle.set_w(w)

        X_START = self.x_cut - self.endmill.r_c + W
        if X_START > self.X_END:
            if auto_layer:
                log.info("Hit end of travel, polishing off remaining bits.")
                # just finish off the rest of the layer
                self.machine.rapid({'x': self.X_END, 'y': self.Y_START})
                self.machine.rapid({'z': self.cut_z})
                self.machine.cut({'y': self.Y_END}, self.f_r_clearing)
                self.machine.rapid({'z': self.cut_z + self.D + 1e-3})
                self.machine.hold_until_still()
                
                # start next layer
                self.begin_layer(self.D)

                log.info("Actually performing cut now.")
                # try again, return result
                return self.cut(conditions, save = save, auto_layer=False)
            else:
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
        time_start = time.perf_counter()
        self.spindle.start_measurement(outQueue, time_start)
        self.tfd.start_measurement(outQueue, time_start)
        self.machine.hold_until_condition(
            lambda state: state['wpos']['y'] > self.Y_DATA_END)

        log.info("Ending data collection")
        self.spindle.stop_measurement()
        self.tfd.stop_measurement()

        log.info("Finished cut, collected " +
                 str(outQueue.qsize()) + " data points.")

        self.machine.rapid({'z': self.cut_z + self.D + 1e-3})
        self.machine.rapid({'y': self.Y_START})
        self.machine.hold_until_still()

        self.x_cut += W

        Ts, Fys = self.dump_queue(outQueue)

        data = Data(self.D, W, f_r, w, self.endmill, Ts, Fys)
        if self.save_as and save:
            with shelve.open(os.path.join("saved_cuts", "db")) as db:
                if self.save_as in db:
                    existing = db[self.save_as]
                    existing.append(data)
                    db[self.save_as] = existing
                else:
                    db[self.save_as] = [data]
                
                db.sync()

            log.info("Data saved under name " + self.save_as)
        return data

    def close(self):
        log.info("Returning home before closing")
        self.machine.rapid({'x': 0, 'y': 0, 'z': 0})
        time.sleep(0.1)
        self.machine.hold_until_still()

    def dump_queue(self, outQueue):
        Ts, Fys = list(), list()
        while not outQueue.empty():
            t, datum = outQueue.get()
            datum_type, datum_value = datum
            if datum_type == 'spindle':
                Ts.append([t, datum_value])
            elif datum_type == 'tfd':
                Fys.append([t, datum_value])

        return np.array(Ts), np.array(Fys)


if __name__ == "__main__":
    log.info("Hi")
    
