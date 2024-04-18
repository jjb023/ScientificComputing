

def check_ode_inputs(f, x0, maxstep, *params):
    """
    This function checks the inputs of a function.
    
    Parameters
    ----------
    f : function
        The function representing the ODE system.
    x0 : float
        The current state of the system.
    maxstep : float
        The maximum step size.
    * params : tuple
        Additional parameters.
        
    Returns:
    ----------
    None
        """
        
    if not callable(f):
        raise TypeError("Input function is not callable")
    
    if not isinstance(x0, (int, float)):
        raise TypeError("Initial state must be a number")
    
    if not isinstance(maxstep, (int, float)):
        raise TypeError("Maximum step size must be a number")
    
    if maxstep <= 0:
        raise ValueError("Maximum step size must be positive")
    
    if not isinstance(params, tuple):
        raise TypeError("Additional parameters must be a tuple")