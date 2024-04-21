from New import *
import unittest
import numpy as np

# Test different inputs to make sure they are the correct format for all functions within New.py
def testinputs():
    failed_tests = []
    def f_ddot(X,t):
        x, y = X
        dxdt = y
        dydt = -x
        return np.array([dxdt, dydt])
    
    # Test 1: Correct input
    try:
        x0 = np.array([1, 0])  # Initial conditions
        t_eval = np.linspace(0, 10, 101)  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode(f_ddot, x0, t_eval, max_step, method, system)
        assert X.shape == (2, 101)
    except:
        failed_tests.append("Test 1 failed ")
    else:
        print("Test 1 passed")


    # Test 2: Wrong type of function
    try:
        x0 = np.array([1, 0])  # Initial conditions
        t_eval = np.linspace(0, 10, 101)  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode("f_ddot", x0, t_eval, max_step, method, system)
        assert False
    except TypeError:
        failed_tests.append("Test 2 passed")


    # Test 3: ODE has incorrect output
    def f_wrong_output(x, t):
        return x

    try:
        x0 = np.array([1, 0])  # Initial conditions
        t_eval = np.linspace(0, 10, 101)  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode(f_wrong_output, x0, t_eval, max_step, method, system)
        assert False
    except AssertionError:
        failed_tests.append("Test 3 passed")


    # Test 4: ODE outputs wrong size
    def f_wrong_size(x, t):
        return np.array([x[0]])

    try:
        x0 = np.array([1, 0])  # Initial conditions
        t_eval = np.linspace(0, 10, 101)  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode(f_wrong_size, x0, t_eval, max_step, method, system)
        assert False
    except AssertionError:
        failed_tests.append("Test 4 passed")


    # Test 5: x0 is wrong type and size
    try:
        x0 = 1  # Initial conditions
        t_eval = np.linspace(0, 10, 101)  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode(f_ddot, x0, t_eval, max_step, method, system)
        assert False
    except TypeError:
        failed_tests.append("Test 5 passed")


    # Test 6: t_eval is wrong type and size
    try:

        x0 = np.array([1, 0])  # Initial conditions
        t_eval = 10  # Evaluation points
        max_step = 0.1  # Maximum step size
        method = 'RK4'  # Numerical method
        system = True  # System of ODEs
        X = solve_ode(f_ddot, x0, t_eval, max_step, method, system)
        assert False
    except TypeError:
        failed_tests.append("Test 6 passed")

    
    if len(failed_tests) == 0:
        print('\n---------------------------------------\n')
        print("All tests passed!")
        print('\n---------------------------------------\n')
    else:
        print('Some input tests failed:')
        for test in failed_tests:
            print('\n---------------------------------------\n')
            print(test)
            print('\n---------------------------------------\n')

if __name__ == '__main__':
    testinputs()
    unittest.main()
