import numpy as np


# Predator Prey Equations
def predator_prey(X, t, params):
    """
    Compute the time derivative of the predator-prey system.
    
    Parameters
    ----------
    X : array
        Initial conditions.
    t : float
        The evaluation time.
    params : tuple
        The parameters of the predator-prey system.
        
    Returns
    -------
    dXdt : array
        The time derivative of the predator-prey system.
    """
    print(f"Received X: {X}, type: {type(X)}")
    if not isinstance(X, (list, tuple, np.ndarray)):
        raise TypeError("X must be a list, tuple, or numpy array")
    x, y = X
    a, b, d = params
    dxdt = x*(1-x) - (a*x*y) / (d+x)
    dydt = b*y*(1-(y/x))
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