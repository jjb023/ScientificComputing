import unittest
import numpy as np 
from shooting import shooting

class TestShootingMethod(unittest.TestCase):
    def test_func(self):
        # Define a simple ODE function and phase condition
        def ode_function(t, y):
            return y
        def phase_condition(sol, u0, *params):
            return sol - u0

        # Define the initial guess and period
        u0 = np.array([1.0])
        T = 1.0

        # Get the function to test
        func = shooting(ode_function, phase_condition)

        # Test the function
        result = func(u0, T, ())
        expected = np.array([0.0])  # For this simple ODE, the solution should be the same as the initial guess
        np.testing.assert_allclose(result, expected)

if __name__ == '__main__':
    unittest.main()