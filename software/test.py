from sensors import Machine, Spindle, TFD
import time
import unittest


class TestMachine(unittest.TestCase):
    PORT = '/dev/ttyS25'

    def testMovement(self):
        machine = Machine(TestMachine.PORT)
        machine.unlock()
        machine.zero()
        machine.rapid(
            {
                'X' : 10,
                'Y' : 10,
            }
        )
        machine.cut(
            {
                'X' : 0,
                'Y' : 0,
            },
            100
        )
    def testState(self):
        machine = Machine(TestMachine.PORT)
        machine.unlock()
        state, coords, _ = machine.get_state()
        self.assertEqual(state, "Idle")
        self.assertEqual(coords, {'x':0, 'y': 0, 'z': 0})
        

# class TestTFD(unittest.TestCase):
#     PORT = '/dev/ttyS6'

#     def testForce(self):
#         tfd = TFD(TestTFD.PORT)
#         force = tfd.get_force()
#         self.assertNotEqual(force, 0)

if __name__ == '__main__':
    unittest.main()