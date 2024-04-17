import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from ODEs import solve_ode

def shooting(u0, function, phase_condition, args, method='RK4', tmax=25, dt=0.01):
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
    method : string
        The method to use, either 'Euler' or 'RK4'.
    tmax : float
        The final time.
    dt : float
        The time step size.
        
    Returns
    -------
    u : array
        The solution to the ODE system that meets the phase condition.
        """
    
    # Ensure u0 is an array
    u0 = np.asarray(u0)

    def objective(u0):
        # Solve ODE system
        u, t = solve_ode(function, u0, 0, dt, tmax, method=method)
        return phase_condition(u[-1], u0, *args)
    
    # Find root of objective function
    result = root(objective, u0, method='hybr')

    if result.success:
        u, t = solve_ode(function, result.x, 0, dt, tmax, method)
        return u, t
    else:
        raise ValueError("Root finding didn't converge.")
    

    
    