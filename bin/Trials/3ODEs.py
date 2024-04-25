import numpy as np 
import math
import matplotlib.pyplot as plt
from bin.inputchecks import *
from Trials.examplefuncs import *

# Euler Step Method
def euler_step(f, x, t, dt, *params):
    """
    Perform a single Euler step.

    Parameters
    ----------
    f : function
        The function representing the ODE system.
    x : float
        ODE solution at t.
    t : float
        The time to evaluate.
    dt : float 
        The step size.
    * params : tuple
        Additional parameters.
        
    Returns:
    ----------
    xnew : float
        The ODE solution at t + dt.
        """
    
    checkf(f)
    checkx(x)
    checkt(t)
    checkdt(dt)
    checkparams(params)
    
    x1 = x + dt * f(x, t, *params)
    t1 = t + dt
    return x1, t1

# RK4 Method
def RK4_step(f, x, t, dt, *params):
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
    
    checkf(f)
    checkx(x)
    checkt(t)
    checkdt(dt)
    checkparams(params)
    
    k1 = f(t, x, *params)
    k2 = f(t + 0.5*dt, x + 0.5*dt*k1, *params)
    k3 = f(t + 0.5*dt, x + 0.5*dt*k2, *params)
    k4 = f(t + dt, x + dt*k3, *params)
    x1 = np.array([x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)])
    return x1, t + dt



# Solve an ODE or system of ODEs by iterating the method until a specified end time.   
def solve_ode(f, x0, t0, t1, maxstep, method, *params):
    """    Solve an ODE to a given time using either Euler or RK4 method.

    Parameters
    ----------
    f : function
        The function representing the ODE system. (x, t, *params).
    x0 : 
        The initial state of the system scalar or array depending on the system.
    t0 : float
        The initial time.
    t1 : float
        The final time.
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
    
    checkf(f)
    checkmaxstep(maxstep)
    checkmethod(method)
    checkparams(params)

    x = np.array(x0)
    t = t0

    nsteps = math.ceil((t1 - t0) / maxstep)    # number of time steps

    dt = (t1 - t0) / nsteps    # step size

    X = np.zeros((nsteps, len(x0) if isinstance(x0, (list, tuple, np.ndarray)) else 1))     # Initialise the solution
    print (f"X: {X}, type: {type(X)}")
    X[0] = x
    # Iterate the method
    for i in range(nsteps):
        dt = min(dt, t1 - t)    # Adjust the step size
        if method == 'Euler':
            x, t = euler_step(f, x, t, dt, *params)
        elif method == 'RK4':
            x, t = RK4_step(f, x, t, dt, *params)
        X[i+1] = x
    return np.array(X), np.linspace(t0, t1, nsteps)




# Example pred prey
# Initial conditions
x0 = np.array([1, 0])
# Parameters
params = (1.0, 0.2, 0.1)
# Evaluation points
t0 = 0
t1 = 10
# Maximum step size
maxstep = 0.1
# Numerical method
method = 'RK4'

X, t = solve_ode(predator_prey, x0, t0, t1, maxstep, method, params)


