import numpy as np
import math
from examplefuncs import *
import matplotlib.pyplot as plt
from inputchecks import *
from ODEcheck import *
from courseworkfuncs import *


def euler_step(f, x0, t0, dt, *params):
    """
    Single step using the Euler Method at x0, t0.
    
    Parameters
    ----------
        f : function
        The function representing the ODE system to solve.
        x0 : float/list
        The initial state(s) (x) of the system.
        t0 : float
        The initial time.
        dt : float
        The step size.
        *params : array
        Additional parameters.
        
    Returns
    ----------
        x1 : float/list
        The state (x) of the system after a single Euler step.
        t1 : float
        The time after a single Euler step.
    """
    k1 = f(t0, x0, *params)
    x1 = x0 + dt * k1
    t1 = t0 + dt

    return x1, t1

def RK4_step(f, x0, t0, dt, *params):
    """
    Single step using the RK4 Method at x0, t0.
    
    Parameters
    ----------
        f : function
        The function representing the ODE system to solve.
        x0 : float/list
        The initial state(s) (x) of the system.
        t0 : float
        The initial time.
        dt : float
        The step size.
        *params : array
        Additional parameters.
        
    Returns
    ----------
        x1 : float/list
        The state (x) of the system after a single RK4 step.
        t1 : float
        The time after a single RK4 step.
    """
    k1 = f(x0, t0, *params)
    k2 = f(x0 + 0.5*dt*k1, t0 + 0.5*dt, *params)
    k3 = f(x0 + 0.5*dt*k2, t0 + 0.5*dt, *params)
    k4 = f(x0 + dt*k3, t0 + dt, *params)
    x1 = x0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    t1 = t0 + dt

    return x1, t1

def solve_to(f, x1, t1, t2, dtmax, method='RK4', *params):
    """
    Solve an ODE system from time t0 to t1 starting from x0, using specified method.
    
    Parameters
    ----------
    f : function
        The function representing the ODE system to solve.
    x1 : float/list
        The initial state(s) (x) of the system.
    t1 : float
        The initial time.
    t2 : float
        The final time.
    dtmax : float
        The maximum step size.
    method : string
        The method to use, either 'Euler' or 'RK4'.
    *params : array
        Additional parameters."""
    
    methods = {'Euler': euler_step, 'RK4': RK4_step}
    
    minsteps = math.floor((t2-t1)/dtmax)
    

    for i in range(minsteps):
        x1, t1 = methods[method](f, x1, t1, dtmax, *params)

    if t1 < t2:
        x1, t1 = methods[method](f, x1, t1, t2-t1, *params)

    return x1

def solve_ode(f, x0, t0, t1, dtmax, method, system, *params):
    """
    Solve an ODE to a given time using specified method.

    Parameters
    ----------
    f : function
        The function representing the ODE system.    
    x0 : float/list
        The initial state(s) (x) of the system.
    t0 : float
        The initial time.
    t1 : float
        The final time.
    dtmax : float
        The maximum step size.
    method : string
        The method to use, either 'Euler' or 'RK4'.
    system : boolean
        If the ODE is a system or not.
    *params : array
        Additional parameters for the method.

    Returns
    ----------
    X : list
        The state of the system after the final time.
    T : float
        The final time.
    """



    steps = math.ceil((t1-t0)/dtmax)

    if system:
        X = np.zeros((steps+1, len(x0)))
    else:
        X = np.zeros(steps+1)
    
    T = np.zeros(steps+1)

    X[0] = x0
    T[0] = t0

    for i in range(steps):
        if T[i] + dtmax < t1:
            T[i+1] = T[i] + dtmax
        else:
            T[i+1] = t1

        if system:
            X[i+1] = solve_to(f, X[i], T[i], T[i+1], dtmax, method, *params)
        else:
            X[i+1] = solve_to(f, X[i], T[i], T[i+1], dtmax, method, *params)

    if system:
        X = X.transpose()
        
    return X, T
    



def main():
    # Solve ODE using Euler and plot
    f = fdot
    x0 = 1
    t0 = 0 
    t1 = 10
    dtmax = 0.1
    method = 'Euler'
    system = False
    params = ()
    
    fdoteuler, teuler = solve_ode(f, x0, t0, t1, dtmax, method, system, params)
    plt.plot(teuler, fdoteuler)
    plt.show()

    # Solve ODE using RK4 and plot
    method = 'RK4'
    fdotrk4, trk4 = solve_ode(f, x0, t0, t1, dtmax, method, system, params)
    plt.plot(trk4, fdotrk4)
    plt.show()

    # Solve ODE system using Euler and plot
    f = fddot
    x0 = [1, 0]
    t0 = 0
    t1 = 10
    dtmax = 0.1
    method = 'Euler'
    system = True
    params = ()

    fddoteuler, teuler = solve_ode(f, x0, t0, t1, dtmax, method, system, params)
    plt.plot(teuler, fddoteuler)
    plt.show()

    # Solve ODE system using RK4 and plot
    method = 'RK4'
    fddotrk4, trk4 = solve_ode(f, x0, t0, t1, dtmax, method, system, params)
    plt.plot(trk4, fddotrk4)
    plt.show()

if __name__ == "__main__":
    main()
    
    