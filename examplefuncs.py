import numpy as np

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





def phase_condition(X0, **params):
    """
    Phase condition for the predator-prey system. dx/dt = 0 at t = 0.
    
    Parameters
    ----------
    X0 : array
        Initial conditions.
    params : dict
        Additional parameters.
        
    Returns
    -------
    float
        Gradient at t = 0.
    """
    dxdt = predator_prey(X0, 0, params)
    return dxdt[0]

def system_derivs(x, y):
    return [y, -x]  # dx/dt = y, dy/dt = -x