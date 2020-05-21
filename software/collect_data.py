from sensors import Machine, Spindle, TFD
import numpy as np
import time
import shelve
import logging
from queue import PriorityQueue
from tqdm import tqdm

class RawDataPacket():
    def __init__(self, D, W, R, N, f_r, w, measurements):
        self.D = D
        self.W = W
        self.R = R
        self.N = N
        self.f_r = f_r
        self.w = w
        self.measurements = measurements

    def decompose_packet_sorta(self):
        tTs, tFys = list(), list()
        Ts, Fys = list(), list()
        for measurement in self.measurements:
            t, reading = measurement
            reading_type, = reading.keys()
            reading_data, = reading.values()
            if reading_type == 'spindle':
                tTs.append(t)
                Ts.append(reading_data['torque'])
            elif reading_type == 'tfd':
                tFys.append(t)
                Fys.append(reading_data)
        return (tTs, tFys, Ts, Fys)
        
    def decompose_packet(self):
        Ts, Fys = list(), list()
        for measurement in self.measurements:
            _, reading = measurement
            reading_type, = reading.keys()
            reading_data, = reading.values()
            if reading_type == 'spindle':
                Ts.append(reading_data['torque'])
            elif reading_type == 'tfd':
                Fys.append(reading_data)

        T = np.median(Ts)
        Fy = np.median(Fys)
        return DataPoint(self.D, self.W, self.R, self.N, self.f_r, self.w, T, Fy) 


class MillingTest():
    def __init__(self, machine_port, spindle_port, tfd_port):
        self._logger = logging.Logger("Milling Test")
        self._logger.setLevel(logging.INFO)
        self.machine = Machine(machine_port)
        self.spindle = Spindle(spindle_port)
        self.tfd = TFD(tfd_port)
        print("Initialized all serial objects")

        # initialize machine
        self.machine.unlock()
        self.machine.zero()
        print("Machine unlocked and zeroed.")

        # create test dbs
        self.db = shelve.open('results')

    def run_test(self, x, y, D, W, R, fmin, fmax, w, N, test_name, offset_depth = 0):
        # save params


        # calculate constants
        CLEARANCE = R
        Y_START = -CLEARANCE * 1.5
        Y_END = CLEARANCE * 1.5 + y
        SWEEPS = int((x - R * 2) / W)
        Y_DATA_START = CLEARANCE
        Y_DATA_END = y - CLEARANCE

        # calculate arrays and whatnot
        feeds = np.linspace(fmin, fmax, SWEEPS)
        xpos = np.linspace(CLEARANCE + W, CLEARANCE + W * SWEEPS, SWEEPS)

        # start spindle
        self.spindle.set_w(w)
        print("Spindle started")

        # initial clearing cycle 
        self.machine.rapid({'x': CLEARANCE, 'y': Y_START, 'z': 1})
        self.machine.rapid({'z': -(D + offset_depth)})
        self.machine.cut({'y': Y_END}, (fmin + fmax)/2)
        print("Clearing cycle sent")
        self.machine.hold_until_still()

        print("Starting actual cuts. " + str(SWEEPS) + " cuts in queue.")
        # begin collecting data and cutting
        results = []
        for f_r, x in tqdm(zip(feeds, xpos)):
            # position
            self.machine.rapid({'z': 1})
            self.machine.rapid({'x': x, 'y': Y_START})
            self.machine.rapid({'z': -(D + offset_depth)})
            self.machine.hold_until_still()
            print("Positioned for cut at x = " + str(x) + " with feed " + str(f_r))
            # measure spindle 
            self.spindle.find_avg_current()
            # tare 
            self.tfd.tare()
            print("Current calibrated, tfd re-tared")
            # send cut command
            self.machine.cut({'y': Y_END}, f_r)
            # delay until we get to a significant place
            print("Waiting until fully in cut")
            self.machine.hold_until_condition(lambda state : state['wpos']['y'] > Y_DATA_START)
            # start counting time
            outQueue = PriorityQueue()
            start = time.perf_counter()
            self.machine.start_measurement(outQueue, start)
            self.spindle.start_measurement(outQueue, start)
            self.tfd.start_measurement(outQueue, start)
            # fetch all data until finished
            print("Measurement started, waiting until cut is left")
            self.machine.hold_until_condition(lambda state : state['wpos']['y'] > Y_DATA_END)
            print("Cut finished")
            # stop measurements
            self.machine.stop_measurement()
            self.spindle.stop_measurement()
            self.tfd.stop_measurement()
            # dump queue
            data = []
            print("Dumping data from queue")
            while not outQueue.empty():
                data.append(outQueue.get())
            print("Queue dumped")
            # add result
            result = RawDataPacket(D * 1e-3, W * 1e-3, R * 1e-3, N, f_r * 1.6667e-5, w, data)
            results.append(result)
            print("Cycle finished.")

        # return home
        print("Finished with all cycles")
        self.machine.rapid({'z': 1})
        self.machine.rapid({'x': 0, 'y': 0})
        self.machine.rapid({'z': 0})

        self.machine.hold_until_still()

        self.db[test_name] = results 
        
        self.db.sync()

    def close(self):
        self.spindle.close()
        self.db.close()



if __name__ == "__main__":
    test = MillingTest('/dev/ttyS25', '/dev/ttyS5', '/dev/ttyS26')
    test.run_test(40, 50.8, 0.5, 3.175, 6.35 / 2, 499, 500, 500, 3, 'fake', 0)
    test.run_test(40, 50.8, 1, 3.175 / 4, 6.35 / 2, 100, 600, 500, 3, '4', 0.5)
    test.run_test(39.9, 50.8, 1, 3.175 / 2, 6.35 / 2, 100, 600, 500, 3, '5', 1.5)
    test.run_test(35.8, 50.8, 1, 3.175 / 1, 6.35 / 2, 100, 600, 500, 3, '6', 2.5)
    test.close()




                