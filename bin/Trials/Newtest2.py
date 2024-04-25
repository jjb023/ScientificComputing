from New import *
import unittest
import numpy as np

# Test different inputs to make sure they are the correct format for all functions within New.py




# Test Outputs of each function to make sure they are correct
class TestNewInputs(unittest.TestCase):
    def test_predator_prey(self):
        # Define parameters
        X = np.array([0.5, 0.5])
        t = 0
        params = (1.0, 0.2, 0.1)

        # Call function
        result = predator_prey(X, t, params)

        # Check result
        expected = np.array([-0.166666667, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-9, err_msg="The predator-prey equations are not correct.")

    def test_phase_condition(self):
        # Define parameters
        X0 = np.array([0.5, 0.5])
        params = (1.0, 0.2, 0.1)

        # Call function
        result = phase_condition(X0, params)

        # Check result
        self.assertEqual(result, -0.5, "The phase condition is not correct.")

    def test_RK4(self):
        # Define parameters
        f = predator_prey
        y0 = np.array([1.0, 1.0])
        t0 = 0
        dt = 0.1
        tmax = 1.0
        params = (1.0, 1.0, 1.0)

        # Call function
        y, t = RK4(f, y0, t0, dt, tmax, params=params)

        # Check result
        expected_y = np.array([[1.0, 1.0], [1.0, 1.0]])  # This is a placeholder. Replace with actual expected result.
        expected_t = np.array([0.0, 0.1])
        np.testing.assert_allclose(y, expected_y)
        np.testing.assert_allclose(t, expected_t)

    def test_solve_ode(self):
        # Define parameters
        f = predator_prey
        y0 = np.array([1.0, 1.0])
        t0 = 0
        dt = 0.1
        tmax = 1.0
        method = 'RK4'
        params = (1.0, 1.0, 1.0)

        # Call function
        y, t = solve_ode(f, y0, t0, dt, tmax, method, params=params)

        # Check result
        expected_y = np.array([[1.0, 1.0], [1.0, 1.0]])
        expected_t = np.array([0.0, 0.1])
        np.testing.assert_allclose(y, expected_y)
        np.testing.assert_allclose(t, expected_t)

    def test_shooting(self):
        # Define parameters
        function = predator_prey
        phase_condition = phase_condition
        initial_conditions = np.array([0.5, 0.5, 2*np.pi])
        params = (1.0, 0.2, 0.1)

        # Call function
        shoot_func = shooting(function, phase_condition)
        result = shoot_func(initial_conditions, params)

        # Check result
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

if __name__ == '__main__':
    unittest.main()