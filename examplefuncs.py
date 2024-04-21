import numpy as np

def fdot(x, t, *params):
    # Single ODE
    return np.array([x])

def fdotsol(x, t):
    # Solution to ODE
    return np.exp(t)

def fddot(u, t, *params):
    # System of ODEs
    x, y = u
    dxdt = y
    dydt = -x
    dXdt = np.array([dxdt, dydt])
    return dXdt

def fddotsol(t):
    # Solution to system of ODEs
    x = np.cos(t) + np.sin(t)
    y = np.cos(t) - np.sin(t)
    return np.array([x, y])


def predator_prey(X, t, params):
    """
    Compute the time derivative of the predator-prey system.
    
    Parameters
    ----------
    X : array
        State vector [prey population, predator population].
    t : float
        The evaluation time (not used in this function but included for compatibility).
    params : tuple
        The parameters of the predator-prey system (intrinsic growth rate of prey,
        predation rate constant, predator efficiency, intrinsic death rate of predators).
        
    Returns
    -------
    dXdt : array
        The time derivative of the predator-prey system.
    """
    
    if not isinstance(X, (list, tuple, np.ndarray)):
        raise TypeError("X must be a list, tuple, or numpy array")
    
    x, y = X  # Prey population, Predator population
    a, b, d = params  # Unpack parameters
    
    dxdt = x * (1 - x) - (a * x * y) / (d + x)  # Change in prey population
    dydt = b * y * (1 - (y / x))  # Change in predator population
    
    return np.array([dxdt, dydt])

