import serial
import pyvesc
import time
import numpy as np
import shelve
import threading
import logging
import queue

from abc import ABC, abstractmethod

class Sensor(ABC):
    def __init__(self):
        self.sensorThread = None
        self.measure = threading.Event()

    @abstractmethod
    def sensorFunction(self):
        ...


    def start_measurement(self, outQueue, initialTime):
        self.measure.set()
        self.sensorThread = threading.Thread(target=self.measureLoop,
                                                   daemon=True,
                                                   args=(outQueue, initialTime))
        self.sensorThread.start()

    def measureLoop(self, outQueue, initialTime):
        while self.measure.is_set():
            measurement = self.sensorFunction()
            t = time.perf_counter() - initialTime
            outQueue.put((t, measurement))
                

    def stop_measurement(self):
        if not self.sensorThread:
            raise Exception("Measurement not started yet")
        self.measure.clear()
        self.sensorThread.join()

class Machine(Sensor):
    """
    Initialize Machine interface.
    Args:
        port: port to use
    """
    def __init__(self, port : serial.Serial):
        super().__init__()
        self._logger = logging.Logger("Machine")
        self._logger.setLevel(logging.DEBUG)
        self.port = serial.Serial(port, 115200, 
                                  parity=serial.PARITY_NONE, 
                                  stopbits=serial.STOPBITS_ONE, 
                                  bytesize=serial.EIGHTBITS,
                                  timeout=0.5, 
                                  writeTimeout=0)

        self.port_lock = threading.RLock()
        self.last_position = (0,0,0)


        self._logger.info("Waking up GRBL")
        self.write("")
        self.write("")
        time.sleep(2)
        self.port.reset_input_buffer()
        self.port.reset_output_buffer()
        self._logger.info("GRBL woken up")
        
    def __del__(self):
        self.write_ok("!")
        self.port.close()
    
    def write(self, msg):
        """Writes a message to the serial port"""
        self.port_lock.acquire()
        self.port.write((msg.strip() + "\r\n").encode('ascii'))
        self.port.flush()
        self.port_lock.release()

    def read(self):
        """Reads a message from GRBL"""
        self.port_lock.acquire()
        resp = self.port.readline().decode('ascii').strip()
        self.port_lock.release()
        return resp

    def write_ok(self, msg : str, attempts = float('inf')):
        """
        Writes a message to GRBL, blocks until response acknowledged
        Args:
            msg: message to write.
            attempts: number of times an "ok" response is checked for.
        """
        self.port_lock.acquire()
        self.write(msg)
        while attempts > 0:
            resp = self.read()
            if "ok" in resp:
                self.port_lock.release()
                return
            elif "error" in resp:
                raise Exception("Error thrown by GRBL: " + resp)
            attempts -= 1
        raise Exception("No OK response received.")

    def read_state(self):
        """Attempts to read state, will block until it receives the state message"""
        self.port_lock.acquire()
        self.write("?")
        while True:
            resp = self.read()
            if "<" in resp:
                self.port_lock.release()
                return resp
    
    def home(self):
        self.write_ok('$H', 5)

    def unlock(self):
        self.write_ok("$X", 5)
        self._logger.info("Machine unlocked.")

    def zero(self):
        self.write_ok("G10 P0 L20 X0 Y0 Z0", 5)
        self._logger.info("Machine zero set.")

    
    def rapid(self, coords):
        self.write_ok("G0 " + " ".join([key + str(value) for key, value in coords.items()]))
        self._logger.info("Rapid move to " + str(coords))

    def cut(self, coords, feed):
        self.write_ok("G1 " + " ".join([key + str(value) for key, value in coords.items()]) + " F" + str(feed))
        self._logger.info("Cut move to " + str(coords))

    def get_state(self):
        resp = self.read_state()
        msgs = resp.strip("<>").split("|")
        state = msgs[0]
        others = msgs[1:]
        reports = {other.split(":")[0]:other.split(":")[1] for other in others}
        x, y, z = [float(i) for i in reports["WPos"].split(",")]
        feed, _ = reports["FS"].split(",")
        dx, dy, dz = [cur - last for cur, last in zip((x, y, z), self.last_position)]
        self.last_position = (x, y, z)
        return {'state': state, 'wpos':{'x': x, 'y': y, 'z': z}, 'feed': feed, 'direction': {'dx': dx, 'dy': dy, 'dz': dz}}

    def hold_until_still(self):
        self._logger.info("Holding until Idle")
        while (self.get_state()['state'] != "Idle"):
            time.sleep(0.1)
        self._logger.info("Idle detected")

    def hold_until_condition(self, condition):
        self._logger.info("Holding until condition met")
        time.sleep(0.1)
        while (not condition(self.get_state())):
            time.sleep(0.1)
        self._logger.info("Condition met")


    def sensorFunction(self):
        return ('machine', self.get_state())


class Spindle(Sensor):
    REDUCTION = 4

    def __init__(self, port):
        super().__init__()
        self._logger = logging.Logger("Spindle")
        self._logger.setLevel(logging.DEBUG)
        self._logger.info("Waking up VESC")
        self.spindle = pyvesc.VESC(port)
        self._logger.info("VESC Ready")
        self.spindle_lock = threading.Lock()

        self.avg_current = 0

    def close(self):
        self.spindle_lock.acquire()
        self.spindle.set_rpm(0)
        self.spindle.stop_heartbeat()
        self.spindle_lock.release()

    def find_avg_current(self):
        self.avg_current = 0
        self.spindle_lock.acquire()
        time.sleep(1)
        samples = 300
        for _ in range(samples):
            time.sleep(0.01)
            measurements = self.spindle.get_measurements()
            self.avg_current += measurements.avg_motor_current

        self.spindle_lock.release()
        self.avg_current /= samples


    def set_w(self, w):
        self.spindle_lock.acquire()
        rpm = int(w * (60 / (2 * np.pi))) * 4
        self.spindle.set_rpm(rpm)
        time.sleep(1)
        confirmation = self.spindle.get_measurements()
        self.spindle_lock.release()
        self._logger.info("Attempted to set RPM of " + str(rpm) + ", actual RPM is " + str(confirmation.rpm))
        if rpm > 0 and confirmation.rpm < rpm * 0.75:
            self.spindle.set_rpm(0)
            raise Exception("Spindle did not approach requested speed of " + str(rpm) + ", actual speed is " + str(confirmation.rpm))

    def sensorFunction(self):
        self.spindle_lock.acquire()
        measurements = self.spindle.get_measurements()
        self.spindle_lock.release()
        all_measurements = {attr: getattr(measurements, attr) for attr in dir(measurements) if (attr[0] != '_' and "fields" not in attr)}
        all_measurements['avg_current'] = self.avg_current
        all_measurements['torque'] = (measurements.avg_motor_current - self.avg_current) * 0.0348134
        return ('spindle', all_measurements)

class TFD(Sensor):

    # completely insane strategy:
    """
    0 bits to 0 kg
    677868.0 bits to one kg
    4122187.0 bits to five kg
    5050401.0 bits to 6 kg
    linear fit through all points leads to conversion value of 832116 bits to one kg
    hella inaccurate but eh
    cubic equation from kg to bits:
    2781.62 * x ** 3 + 745886 * x
    unknown if it works past 6kg but eh it's a start lol
    https://bit.ly/3c6lR2M for fitting memes
    """

    BITS_TO_N = 227753.0 / 9.81

    def __init__(self, port):
        super().__init__()
        self.port = serial.Serial(port, 115200, timeout=0.5)
        self._logger = logging.Logger("TFD")
        self._logger.setLevel(logging.DEBUG)
        self._logger.info("Waking up TFD")
        self.port.write("F".encode('ascii'))
        self.port.flush()
        time.sleep(10)
        self.port.reset_input_buffer()
        self._logger.info("TFD Ready")


    def get_force(self):
        self.port.write("F".encode('ascii'))
        self.port.flush()
        resp = self.port.readline().decode('ascii').strip()
        return float(resp) / TFD.BITS_TO_N


    def calibrate(self):
        self.port.write("T".encode('ascii'))
        self.port.flush()
        while True:
            if "TARED" in self.port.readline().decode('ascii'):
                return


    def sensorFunction(self):
        return ('tfd', self.get_force())

class Spindle_Applied(Sensor):

    RS485_ADDRESS = 0
    I_TO_T = 1
    CALIBRATION_POINTS = 100

    def __init__(self, port):
        self.port = serial.Serial(port, 38400, timeout=0.5)
        self.torque_loss = 0

        self.port_lock = threading.RLock()
        self.port.reset_input_buffer()
        self.port.reset_output_buffer()
        
    def __del__(self):
        self.write_ack("SJ")
        self.port.close()

    def write_ack(self, msg):
        self.port_lock.acquire()
        full_msg = Spindle_Applied.RS485_ADDRESS +\
                   msg +\
                   "\r"
        self.port.write(full_msg.encode('ascii'))
        self.port.flush()
        self.port_lock.release()
        response = self.port.read().decode('ascii')
        if response in ('%', '*'):
            return
        elif response == '?':
            raise Exception("Error sent back from drive.")
        elif not response:
            raise Exception("Command was not understood by drive")
        else:
            raise Exception("Unknown ack received: " + response)

    def write_get_response(self, msg):
        self.port_lock.acquire()
        self.write_ack(msg)
        response = self.port.readline().decode('ascii')
        self.port_lock.release()
        assert response[0] == Spindle_Applied.RS485_ADDRESS , "Address of response incorrect, was actually " + response[0]
        assert response[1:3] == msg[0: 2], "Command of response incorrect, was actually " + response[1:3]
        assert response[4] == "=", "No '=' in response, was actually " + response[4]
        return response[5:]

    def set_w(self, w):
        rpm = int(w * (60 / (2 * np.pi))) 
        self.write_ack("JA50")
        self.write_ack("JL25")
        self.write_ack("CJ")
        time.sleep(1)
        actual_rpm = self.write_get_response("IV")
        if rpm != 0 and actual_rpm < rpm * 0.75:
            self.write_ack("SJ")
            raise Exception("Failed to reach speed of " + str(rpm) + "RPM. Actually reached " + str(actual_rpm) + "RPM.")

    def get_torque(self, calibrated = True):
        I = float(self.write_get_response("IQ"))
        return I * Spindle_Applied.I_TO_T - (self.torque_loss if calibrated else 0)

    def calibrate(self):
        Ts = list()
        for i in range(Spindle_Applied.CALIBRATION_POINTS):
            time.sleep(0.01)
            Ts.append(self.get_torque(calibrated = False))
        
        self.torque_loss = np.median(Ts)

    def sensorFunction(self):
        return ('spindle', self.get_torque())

