import numpy as np



def check_inputs(f, x0, t0, t1, dt, method):
    if not callable(f):
        raise ValueError("Function f must be callable.")
    if not isinstance(x0, (list, np.ndarray)):
        raise ValueError("Initial state x0 must be a list or numpy array.")
    if not (isinstance(t0, (int, float)) and isinstance(t1, (int, float))):
        raise ValueError("Times t0 and t1 must be numbers.")
    if t1 <= t0:
        raise ValueError("End time t1 must be greater than start time t0.")
    if dt <= 0:
        raise ValueError("Time step size dt must be positive.")
    if method not in ['euler', 'rk4', 'midpoint']:
        raise ValueError("Method must be 'euler', 'rk4', or 'midpoint'.")
