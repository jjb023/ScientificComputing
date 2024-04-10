def shooting(u0, function, phase_condition, args):
    """
    Function to find limit cycles of specific ODEs using shooting method.
    
    Parameters
    ----------
    u0 : array
        Initial guess for the solution.
    
    function : function
        The function representing the ODE system.
        
    phase_condition : function
        Phase condition for limit cycle.
    
    args : tuple
        Additional arguments to pass to the function.
    
    Returns
    -------
    u : array
        The solution to the ODE system.

        """
    
    