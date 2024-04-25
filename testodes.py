import unittest
import numpy as np
from odes import euler_step, RK4_step, solve_to, solve_ode

class TestODEFunctions(unittest.TestCase):

    def setUp(self):
        # Define a simple ODE for testing: dx/dt = -x, the solution of which is an exponential decay
        self.test_ode = lambda x, t, **params: -x
        self.initial_x = np.array([1.0, 2.0])
        self.t0 = 0
        self.t1 = 1
        self.dt = 0.01  # Using a smaller step size for more accuracy in numerical methods

    def test_euler_step(self):
        result = euler_step(self.test_ode, self.initial_x, self.t0, self.dt)
        expected = self.initial_x + self.dt * np.array(self.test_ode(self.initial_x, self.t0))
        np.testing.assert_array_almost_equal(result, expected)

    def test_RK4_step(self):
        result = RK4_step(self.test_ode, self.initial_x, self.t0, self.dt)
        # Calculate an expected result using a manual RK4 computation for comparison
        k1 = self.test_ode(self.initial_x, self.t0)
        k2 = self.test_ode(self.initial_x + 0.5 * self.dt * k1, self.t0 + 0.5 * self.dt)
        k3 = self.test_ode(self.initial_x + 0.5 * self.dt * k2, self.t0 + 0.5 * self.dt)
        k4 = self.test_ode(self.initial_x + self.dt * k3, self.t0 + self.dt)
        expected = self.initial_x + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        np.testing.assert_array_almost_equal(result, expected)

    def test_solve_to(self):
        result = solve_to(self.test_ode, self.initial_x, self.t0, self.t1, self.dt, 'RK4')
        # Simple exponential decay check
        expected = np.exp(-self.t1)
        self.assertAlmostEqual(result[-1], expected)

    def test_solve_ode(self):
        time_steps = np.linspace(self.t0, self.t1, 100)
        result = solve_ode(self.test_ode, self.initial_x, time_steps, 'RK4')
        # Check array shapes and final value
        self.assertEqual(result.shape, (len(time_steps), len(self.initial_x)))
        expected_final_value = np.exp(-self.t1)
        np.testing.assert_array_almost_equal(result[-1], np.array([expected_final_value]))

if __name__ == '__main__':
    unittest.main()
