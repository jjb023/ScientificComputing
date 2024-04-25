import numpy as np

def check_ode_inputs(f, x, t, dt):
    if not callable(f):
        raise TypeError("Function f must be callable.")
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x must be a numpy array.")
    if not isinstance(t, (int, float)):
        raise TypeError("Time t must be an integer or float.")
    if not isinstance(dt, (int, float)) or dt <= 0:
        raise TypeError("Step size dt must be a positive integer or float.")
