import numpy as np
from examplefuncs import *
import matplotlib.pyplot as plt
from checks import * 

def euler_step(f, x, t, dt, **kwargs):
    """
    Perform a single Euler step for a system of ODEs.
    
    Parameters
    ----------
    f : function
        The function representing the ODE system, must return an array.
    x : array
        The current state of the system.
    t : float
        The current time.
    dt : float
        The time step size.

    Returns
    ----------
    x1 : array
        The state of the system after a single Euler step.
    t1 : float
        The time after a single Euler step.
    """
    x1 = x + dt * np.array(f(x, t, **kwargs))
    t1 = t + dt
    return x1, t1

def rk4_step(f, x, t, dt, **kwargs):
    """
    Perform a single Runge-Kutta 4 step for a system of ODEs.
    
    Parameters
    ----------
    f : function
        The function representing the ODE system, must return an array.
    x : array
        The current state of the system.
    t : float
        The current time.
    dt : float
        The time step size.

    Returns
    ----------
    x1 : array
        The state of the system after the RK4 step.
    t1 : float
        The time after the RK4 step.
    """
    k1 = np.array(f(x, t, **kwargs))
    k2 = np.array(f(x + 0.5 * dt * k1, t + 0.5 * dt, **kwargs))
    k3 = np.array(f(x + 0.5 * dt * k2, t + 0.5 * dt, **kwargs))
    k4 = np.array(f(x + dt * k3, t + dt, **kwargs))
    x1 = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    t1 = t + dt
    return x1, t1

def midpoint_step(f, x, t, dt, **kwargs):
    """
    Perform a single Midpoint step for a system of ODEs.

    Parameters
    ----------
    f : function
        The function representing the ODE system, must return an array.
    x : array
        The current state of the system.
    t : float
        The current time.
    dt : float
        The time step size.

    Returns
    ----------
    x1 : array
        The state of the system after the Midpoint step.
    t1 : float
        The time after the Midpoint step.
    """
    mid_x = x + 0.5 * dt * np.array(f(x, t, **kwargs))
    mid_t = t + 0.5 * dt
    x1 = x + dt * np.array(f(mid_x, mid_t, **kwargs))
    t1 = t + dt
    return x1, t1

def solve_to(f, x0, t0, t1, dt, method='rk4', **kwargs):
    """
    Solves an ODE system from time t0 to t1 starting from x0, using specified method.
    Now supports systems of ODEs.

    Parameters
    ----------
    f : function
        The function representing the ODE system, must return an array.
    x0 : array
        The initial state of the system.
    t0 : float
        The initial time.
    t1 : float
        The final time.
    dt : float
        The time step size.
    method : str
        The method to use ('euler', 'rk4', 'midpoint').

    Returns
    ----------
    times : array
        Array of times at which the ODE was evaluated.
    values : array
        Array of system states corresponding to each time.
    """
    methods = {'euler': euler_step, 'rk4': rk4_step, 'midpoint': midpoint_step}
    step_function = methods.get(method, rk4_step)

    times = np.arange(t0, t1 + dt, dt)
    values = [x0]
    t = t0
    x = np.array(x0, dtype=float)

    for i in range(1, len(times)):
        x, t = step_function(f, x, t, dt, **kwargs)
        values.append(x)

    return times, np.array(values)

def solve_ode(f, x0, t0, t_end, dt, method='rk4', **kwargs):
    """
    Wrapper function to solve an ODE system from initial value x0 at time t0 to end time t_end.
    Uses the solve_to function with specified method. Supports systems of ODEs.

    Parameters
    ----------
    f : function
        The function representing the ODE system, must return an array.
    x0 : array
        The initial state of the system.
    t0 : float
        The initial time.
    t_end : float
        The end time.
    dt : float
        The time step size.
    method : str
        The method to use for stepping ('euler', 'rk4', 'midpoint').

    Returns
    ----------
    times : array
        Array of times at which the ODE was evaluated.
    values : array
        Array of system states corresponding to each time.
    """
    check_inputs(f, x0, t0, t_end, dt, method)
    return solve_to(f, x0, t0, t_end, dt, method, **kwargs)

# Plot the results of the ODE solver
def plot_ode(times, values, xlabel='t', ylabel='x', title=None):
    """
    Plot the results of an ODE solver.

    Parameters
    ----------
    times : array
        Array of times at which the ODE was evaluated.
    values : array
        Array of system states corresponding to each time.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.
    """


    if title is not None:
        plt.title(title)

    if values.shape[1] == 1:
        plt.plot(times, values, label=ylabel)
    else:
        for i in range(values.shape[1]):
            plt.plot(times, values[:, i], label=f'{ylabel}_{i}')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
