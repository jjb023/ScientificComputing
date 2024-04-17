import unittest
import numpy as np
from shooting import shooting  # Adjust the import path as necessary

def vanderpol(t, Y, mu):
    print(f"Y at vanderpol: {Y}, type: {type(Y)}")  # Debug print to check what Y is
    x, y = Y
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return np.array([dxdt, dydt])

def phase_condition(Yf, Y0, mu):
    x_f, y_f = Yf
    x_0, y_0 = Y0
    return np.array([x_f - x_0, y_f])

class TestShootingMethod(unittest.TestCase):
    def test_vanderpol_oscillator(self):
        u0 = np.array([2.0, 0.0])
        print(f"Initial condition u0: {u0}, type: {type(u0)}")  # Debugging output
        mu = 2
        try:
            result, t = shooting(u0, lambda t, Y: vanderpol(t, Y, mu), phase_condition, (mu,), method='RK4', tmax=25, dt=0.01)
            # Assertions to check the final state
        except Exception as e:
            print(f"Test failed with error: {e}")
            self.fail("Shooting method test encountered an error.")


if __name__ == '__main__':
    unittest.main()

