import serial
import pyvesc
import time
import numpy as np
import shelve
import threading
import logging
import queue
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class Sensor(ABC):
    """
    Represents a sensor used for data collection. Implements a basic set of multithreaded data collection subroutines.
    """
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

    def __init__(self, port: serial.Serial, graceful_shutdown = False):
        super().__init__()
        self._logger = logging.getLogger(__name__ + ".machine")
        self.port = serial.Serial(port, 115200,
                                  parity=serial.PARITY_NONE,
                                  stopbits=serial.STOPBITS_ONE,
                                  bytesize=serial.EIGHTBITS,
                                  timeout=0.5,
                                  writeTimeout=0)
        self.graceful_shutdown = graceful_shutdown

        self.port_lock = threading.RLock()
        self.last_position = (0, 0, 0)

        self._logger.info("Waking up GRBL")
        self.write("")
        self.write("")
        time.sleep(2)
        self.port.reset_input_buffer()
        self.port.reset_output_buffer()
        self._logger.info("GRBL woken up")

    def __del__(self):
        if self.graceful_shutdown:
            self.rapid({'z': 1})
            self.rapid({'x': 0, 'y': 0})
            self.rapid({'z': 0})
            self.hold_until_still()
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

    def write_ok(self, msg: str, attempts=float('inf')):
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
        self.write_ok(
            "G0 " + " ".join([key + str(value * 1e3) for key, value in coords.items()]))
        self._logger.info("Rapid move to " + str(coords))

    def cut(self, coords, feed):
        self.write_ok("G1 " + " ".join([key + str(value * 1e3)
                                        for key, value in coords.items()]) + " F" + str(feed * 60000))
        self._logger.info("Cut move to   " + str(coords))

    def get_state(self):
        resp = self.read_state()
        msgs = resp.strip("<>").split("|")
        state = msgs[0]
        others = msgs[1:]
        reports = {other.split(":")[0]: other.split(":")[1]
                   for other in others}
        x, y, z = [float(i) * 1e-3 for i in reports["WPos"].split(",")]
        feed, _ = reports["FS"].split(",")
        dx, dy, dz = [cur - last for cur,
                      last in zip((x, y, z), self.last_position)]
        self.last_position = (x, y, z)
        return {'state': state, 'wpos': {'x': x, 'y': y, 'z': z}, 'feed': float(feed) / 60000, 'direction': {'dx': dx, 'dy': dy, 'dz': dz}}

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
        self._logger = logging.getLogger(__name__ + ".spindle")
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
        self._logger.info("Attempted to set RPM of " + str(rpm) +
                          ", actual RPM is " + str(confirmation.rpm))
        if rpm > 0 and confirmation.rpm < rpm * 0.75:
            self.spindle.set_rpm(0)
            raise Exception("Spindle did not approach requested speed of " +
                            str(rpm) + ", actual speed is " + str(confirmation.rpm))

    def sensorFunction(self):
        self.spindle_lock.acquire()
        measurements = self.spindle.get_measurements()
        self.spindle_lock.release()
        all_measurements = {attr: getattr(measurements, attr) for attr in dir(
            measurements) if (attr[0] != '_' and "fields" not in attr)}
        all_measurements['avg_current'] = self.avg_current
        all_measurements['torque'] = (
            measurements.avg_motor_current - self.avg_current) * 0.0348134
        return ('spindle', all_measurements)


class TFD(Sensor):
    """

    """
    BITS_TO_N = -1714

    def __init__(self, port):
        super().__init__()
        self.port_id = port
        self.port = serial.Serial(port, 115200, timeout=0.5)
        self._logger = logging.getLogger(__name__ + ".tfd")
        self._logger.info("Waking up TFD")
        self.port.write("F".encode('ascii'))
        self.port.flush()
        time.sleep(5)
        self.port.reset_input_buffer()
        self._logger.info("TFD Ready")

    def get_force(self):
        for i in range(5):
            e = "No exception..."
            try:
                self.port.write("F".encode('ascii'))
                self.port.flush()
                resp = self.port.readline().decode('ascii').strip()
                return float(resp) / TFD.BITS_TO_N
            except Exception as ex:
                e = ex
            self._logger.warn("Read #" + str(i + 1) + " failed, trying again... Specific error: " + str(e))
            self.port.close()
            self.port = serial.Serial(self.port_id, 115200, timeout=0.5)
            self.port.reset_input_buffer()
            self.port.reset_output_buffer()
        raise IOError("Failed to get force from TFD")

    def calibrate(self):
        for i in range(5):
            e = "No exception"
            try:
                self.port.write("T".encode('ascii'))
                self.port.flush()
                for _ in range(5):
                    if "TARED" in self.port.readline().decode('ascii'):
                        return

            except Exception as ex:
                e = ex
            # sensor failed to calibrate, retry
            self._logger.warn("Calibrate #" + str(i + 1) + " failed, trying again... Specific error: " + str(e))
            self.port.close()
            self.port = serial.Serial(self.port_id, 115200, timeout=0.5)
            self.port.reset_input_buffer()
            self.port.reset_output_buffer()
            time.sleep(0.1)

        raise IOError("Failed to calibrate TFD")

    def sensorFunction(self):
        return ('tfd', self.get_force())


class Spindle_Applied(Sensor):

    RS485_ADDRESS = 0
    I_TO_T = 0.10281
    CALIBRATION_POINTS = 200

    def __init__(self, port):
        super().__init__()
        self._logger = logging.getLogger(__name__ + ".spindle_applied")
        self._logger.info("Waking up Applied Motion Spindle")
        self.port_id = port
        self.port = serial.Serial(port, 115200, timeout=0.5)
        self.torque_loss = 0
        self.jogging = False

        self.port_lock = threading.RLock()
        self.port.reset_input_buffer()
        self.port.reset_output_buffer()
        self.write_ack("AX")
        self.write_ack("ME")
        self.write_ack("IFD")
        self._logger.info("Applied Motion Spindle Ready")

    def __del__(self):
        if self.jogging:
            self.write("SJ")
        self.write("MD")
        self.port.close()

    def write(self, msg):
        self.port_lock.acquire()
        full_msg = str(Spindle_Applied.RS485_ADDRESS) +\
                   msg +\
                   "\r"
        self.port.write(full_msg.encode('ascii'))
        self.port.flush()
        self.port_lock.release()

    def write_ack(self, msg):
        for i in range(5):
            e = "No exception..."
            try:
                self.port_lock.acquire()
                self.write(msg)
                response = self.port.read_until('\r'.encode('ascii')).decode('ascii').strip()
                self.port_lock.release()
                ack_byte = response[1] if response else None
                if ack_byte in ('%', '*'):
                    return
                elif ack_byte == '?':
                    self._logger.warn("Error sent back from drive.")
                elif not response:
                    self._logger.warn("Command was not understood by drive")
                else:
                    self._logger.warn("Unknown ack received: " + response.strip())
            # trying again
            except Exception as ex:
                e = ex
            self._logger.warn("Write-ack #" + str(i + 1) + " failed, trying again. Specific error: " + str(e))

            # closing and reopening the port to unbork any issues
            self.port_lock.acquire()
            self.port.close()
            self.port = serial.Serial(self.port_id, 115200, timeout=0.5)
            self.port.reset_input_buffer()
            self.port.reset_output_buffer()
            self.write("AR")
            self.write("ME")
            self.write("IFD")
            self.port_lock.release()
            time.sleep(0.1)


        raise IOError("Failed to write command to spindle...")

    def write_get_response(self, msg):
        for i in range(5):
            e = "No Exception..."
            try:
                self.port_lock.acquire()
                self.write(msg)
                response = self.port.read_until("\r".encode('ascii')).decode('ascii').strip()
                self.port_lock.release()
                assert response[0] == str(Spindle_Applied.RS485_ADDRESS), "Address of response incorrect, was actually " + response[0]
                assert response[1:3] == msg[0:2], "Command of response incorrect, was actually " + response[1:3]
                assert response[3] == "=", "No '=' in response, was actually " + response[4]
                return response[4:]

            except Exception as ex:
                e = ex
            self._logger.warn("Write-ack #" + str(i + 1) + " failed, trying again. Specific error: " + str(e))
            self.port_lock.acquire()
            self.port.close()
            self.port = serial.Serial(self.port_id, 115200, timeout=0.5)
            self.port.reset_input_buffer()
            self.port.reset_output_buffer()
            self.write("AR")
            self.write("ME")
            self.write("IFD")
            self.port_lock.release()
            time.sleep(0.1)

        raise IOError("Failed to get proper response from spindle...")

    def set_w(self, w):
        self.port.reset_input_buffer()
        self.port.reset_output_buffer()
        rpm = int(w * (60 / (2 * np.pi)))
        rpm_str = "{:.3f}".format(rpm / 60)
        if rpm:
            if not self.jogging:
                self.write_ack("JS" + rpm_str)
                self.write_ack("DI-1000000")
                self.write_ack("CJ")
                self.jogging = True
            else:
                self.write_ack("DI-1000000")
                self.write_ack("CS-" + rpm_str)
        else:
            self.write_ack("SJ")
            self.jogging = False
            return
        time.sleep(1)
        response = self.write_get_response("IV0")
        actual_rpm = -float(response)
        if actual_rpm < rpm * 0.75:
            self.write_ack("SJ")
            raise Exception("Failed to reach speed of " + str(rpm) +
                            "RPM. Actually reached " + str(actual_rpm) + " RPM.")
        self._logger.info("Set RPM to " + str(rpm) +
                          ", achieved speed of " + str(actual_rpm))

    def get_torque(self, calibrated=True):
        I = -float(self.write_get_response("IC"))
        return I * Spindle_Applied.I_TO_T - (self.torque_loss if calibrated else 0)

    def calibrate(self):
        Ts = list()
        for _ in range(Spindle_Applied.CALIBRATION_POINTS):
            time.sleep(0.01)
            Ts.append(self.get_torque(calibrated=False))

        self.torque_loss = np.median(Ts)
        self._logger.info(
            "Calibrated, torque loss found to be " + str(self.torque_loss))

    def sensorFunction(self):
        return ('spindle', self.get_torque())
