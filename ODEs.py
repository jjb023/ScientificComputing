import numpy as np
import math
import matplotlib.pyplot as plt


# Euler Step Method
def euler_step(f, x0, t0, dt, params):
    """
    Perform a single Euler step.

    Parameters
    ----------
    f : function
        The function representing the ODE system.
    x : array
        The current state of the system.
    t : float
        The current time.
    dt : float
        The time step size.

    Returns:
    ----------
    x1 : array
        The state of the system after a single Euler step.
    t1 : float
        The time after a single Euler step.
    """
    x1 = x0 + dt * f(x0, t0, *params)
    t1 = t0 + dt
    return x1, t1

# RK4 Method
def rk4_step(f, x0, t0, dt, params):
    """
    Perform a single RK4 step.

    Parameters
    ----------
    f : function
        The function representing the ODE system.
    x : array
        The current state of the system.
    t : float
        The current time.
    dt : float
        The time step size.
    
    Returns:
    ----------
    x1 : array
        The state of the system after a single RK4 step.
    t1 : float 
        The time after a single RK4 step.
    """

    # print(f"RK4 input x0: {x0}, type: {type(x0)}")  # Debugging output

    # intermediate values of k1, k2, k3, k4
    k1 = f(x0, t0, *params)
    k2 = f(x0 + 0.5 * dt * k1, t0 + 0.5 * dt, *params)
    k3 = f(x0 + 0.5 * dt * k2, t0 + 0.5 * dt, *params)
    k4 = f(x0 + dt * k3, t0 + dt, *params)
    # final value of x1, t1
    k = 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)
    x1 = x0 + k
    t1 = t0 + dt
    return x1, t1


# Solve to a given time using either method.
def solve_to(f, x0, t0, dt, tmax, method='Euler'):
    """
    Solve an ODE to a given time using either Euler or RK4 method.

    Parameters
    ----------
    f : function
        The function representing the ODE system.
    x0 : array
        The initial state of the system.
    t0 : float
        The initial time.
    dt : float
        The time step size.
    tmax : float
        The final time.
    method : string
        The method to use, either 'Euler' or 'RK4'.

    Returns:
    ----------
    x : array
        The state of the system after the final time.
    t : float
        The final time.
    """
    # Euler method
    if method == 'Euler':
        x, t = x0, t0
        while t < tmax:
            x, t = euler_step(f, x, t, dt)
        return x, t
    # RK4 method
    elif method == 'RK4':
        x, t = x0, t0
        while t < dt:
            x, t = rk4_step(f, x, t, dt)
        return x, t
    else:
        print('Invalid method')
        return None
    
# Solve an ODE or system of ODEs using either method using the solve_to function.
def solve_ode(f, x0, t0, dt, tmax, method='RK4', params=()):
    x, t = np.atleast_1d(x0), t0
    xvals, tvals = [x.copy()], [t]

    step_function = rk4_step if method == 'RK4' else euler_step
    while t < tmax:
        next_t = min(t + dt, tmax)
        actual_dt = next_t - t
        x, t = step_function(f, x, t, actual_dt, params)
        xvals.append(x.copy())
        tvals.append(t)

    return np.array(xvals), np.array(tvals)




# def test():
#     def simple_ode(x, t):
#         return -x
    
#     x0 = 1.0  # Correct initial condition
#     try:
#         solve_ode(simple_ode, x0, 0, 0.01, 1, 'RK4')
#     except Exception as e:
#         print(f"Error: {e}")
    
#     x0 = "wrong type"  # Incorrect initial condition
#     try:
#         solve_ode(simple_ode, x0, 0, 0.01, 1, 'RK4')
#     except Exception as e:
#         print(f"Error: {e}")
    
#     method = "Unknown Method"  # Incorrect method
#     try:
#         solve_ode(simple_ode, 1.0, 0, 0.01, 1, method)
#     except Exception as e:
#         print(f"Error: {e}")

# if __name__ == "__main__":
#     test()

