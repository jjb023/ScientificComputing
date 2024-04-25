# from finalODEs import solve_ode

def odetest():
    """
    Tests for inputs of ODE solver.
    """
    # Test 1: f is not callable
    try:
        solve_ode(1, [1], 0, 1, 0.1)
    except ValueError as e:
        assert str(e) == "Function f must be callable."
    
    # Test 2: x0 is not a list or numpy array
    try:
        solve_ode(lambda x, t: x, 1, 0, 1, 0.1)
    except ValueError as e:
        assert str(e) == "Initial state x0 must be a list or numpy array."
    
    # Test 3: t0 and t1 are not numbers
    try:
        solve_ode(lambda x, t: x, [1], 'a', 1, 0.1)
    except ValueError as e:
        assert str(e) == "Times t0 and t1 must be numbers."
    
    # Test 4: t1 <= t0
    try:
        solve_ode(lambda x, t: x, [1], 1, 0, 0.1)
    except ValueError as e:
        assert str(e) == "End time t1 must be greater than start time t0."
    
    # Test 5: dt <= 0
    try:
        solve_ode(lambda x, t: x, [1], 0, 1, 0)
    except ValueError as e:
        assert str(e) == "Time step size dt must be positive."
    
    # Test 6: method is not 'euler', 'rk4', or 'midpoint'
    try:
        solve_ode(lambda x, t: x, [1], 0, 1, 0.1, method='a')
    except ValueError as e:
        assert str(e) == "Method must be 'euler', 'rk4', or 'midpoint'."
        
    print("All input tests passed.")