from sensors import Machine, Spindle, TFD
from queue import Queue
import time
PORT_1 = '/dev/ttyS26'
PORT_2 = '/dev/ttyS5'

outQueue = Queue()
tfd = TFD(PORT_1)



while True:
    print(tfd.get_force())


# 0.03325758958259111 is positive weight