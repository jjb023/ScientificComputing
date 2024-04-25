import unittest
import numpy as np
from PDEs import xgrid, maxdt, explicit_euler, implicit_euler, diffusionIC

class TestPDEs(unittest.TestCase):

    def test_xgrid(self):
        x, dx, xint = xgrid(10, 0, 1)
        self.assertEqual(len(x), 11)
        self.assertEqual(dx, 0.1)
        self.assertEqual(len(xint), 9)

    def test_maxdt(self):
        dt = maxdt(1, 0.1)
        self.assertAlmostEqual(dt, 0.005, places=7)

    def test_explicit_euler(self):
        u, t = explicit_euler(10, 1, 0, 1, 0, 1, 0.01, 0.1, 1, np.linspace(0, 1, 9), diffusionIC)
        self.assertEqual(u.shape, (101, 9))
        self.assertEqual(len(t), 100)

    def test_implicit_euler(self):
        u, t = implicit_euler(10, 1, 0, 1, 0, 1, 0.01, 0.1, 1, np.linspace(0, 1, 9), diffusionIC)
        self.assertEqual(u.shape, (101, 9))
        self.assertEqual(len(t), 100)

    def test_diffusionIC(self):
        IC = diffusionIC(np.linspace(0, 1, 10), 1)
        self.assertEqual(len(IC), 10)

if __name__ == '__main__':
    unittest.main()