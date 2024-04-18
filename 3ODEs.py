import numpy as np 
import math
import matplotlib.pyplot as plt
from inputchecks import *

# Euler Step Method
def euler_step(f, x0, t0, maxstep, *params):
    """
    Perform a single Euler step.

    Parameters
    ----------
    f : function
        The function representing the ODE system.
    x0 : float
        The current state of the system.
    t0 : float
        The current time.
    maxstep : float 
        The maximum step size.
    * params : tuple
        Additional parameters.
        
    Returns:
    ----------
    x1 : float
        The state of the system after a single Euler step.
    t1 : float
        The time after a single Euler step.
        """
    
    check_ode_inputs(f, x0, maxstep, *params)
    
    x1 = x0 + maxstep * f(x0, t0, *params)
    t1 = t0 + maxstep
    return x1, t1

# RK4 Method
def RK4_step(f, x0, t0, maxstep, *params):
    """
    Perform a single RK4 step.

    Parameters
    ----------
    f : function
        The function representing the ODE system.
    x0 : float
        The current state of the system.
    t0 : float
        The current time.
    maxstep : float 
        The maximum step size.
    * params : tuple
        Additional parameters.
        
    Returns:
    ----------
    x1 : float
        The state of the system after a single RK4 step.
    t1 : float
        The time after a single RK4 step.
        """
    
    check_ode_inputs(f, x0, maxstep, *params)
    
    k1 = f(t0, x0, *params)
    k2 = f(t0 + 0.5*maxstep, x0 + 0.5*maxstep*k1, *params)
    k3 = f(t0 + 0.5*maxstep, x0 + 0.5*maxstep*k2, *params)
    k4 = f(t0 + maxstep, x0 + maxstep*k3, *params)
    x1 = x0 + maxstep/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x1, t0 + maxstep

# Solve to a given time using either method.   
def solve_to(f, x0, t0, maxstep, tmax, method='Euler', *params):
    """
    Solve an ODE to a given time using either Euler or RK4 method.

    Parameters
    ----------
    f : function
        The function representing the ODE system. (x, t, *params).
    x0 : float
        The initial state of the system.
    t0 : float
        The initial time.
    maxstep : float 
        The maximum step size.
    tmax : float
        The final time.
    method : string
        The method to use, either 'Euler' or 'RK4'.
    * params : tuple
        Additional parameters.
        
    Returns:
    ----------
    xvals : list
        The state of the system at each time step.
    tvals : list
        The time at each time step.
        """
    
    check_ode_inputs(f, x0, maxstep, *params)
    
    xvals = [x0]
    tvals = [t0]
    
    while tvals[-1] < tmax:
        if method == 'Euler':
            x1, t1 = euler_step(f, xvals[-1], tvals[-1], maxstep, *params)
        elif method == 'RK4':
            x1, t1 = RK4_step(f, xvals[-1], tvals[-1], maxstep, *params)
        else:
            raise ValueError("Invalid method. Choose either 'Euler' or 'RK4'")
        xvals.append(x1)
        tvals.append(t1)
        
    return xvals, tvals

# Solve an ODE or system of ODEs by iterating the method until a specified end time.   
def solve_ode(f, x0, teval, maxstep, method, system, *params):
    """
    Solve an ODE to a given time using either Euler or RK4 method.

    Parameters
    ----------
    f : function
        The function representing the ODE system. (x, t, *params).
    x0 : 
        The initial state of the system scalar or array depending on the system.
    teval : float
        List of evaluation times.
    maxstep : float 
        The maximum step size to avoid overshooting.
    method : string
        The method to use, either 'Euler' or 'RK4'.
    system : boolean
        If the ODE is a system or not.
    * params : tuple
        Additional parameters for the method.
        
    Returns:
    ----------
    X : list
        The state of the system at the evaluation times. If system is True, then X is a list of arrays.
        """
    
    check_ode_inputs(f, x0, maxstep, *params)
    
    if system:
        X = np.zeros((len(teval), len(x0)))
    else:
        X = np.zeros(len(teval))
    
    X[0] = x0

    for i in range(1, len(teval)):
        to = teval[i]
        to1 = teval[i+1]
        Xsol, tsol = solve_to(f, X[i-1], to, maxstep, to1, method, *params)
        X[i+1] = Xsol

    return X

    