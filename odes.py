import numpy as np
from checks import check_ode_inputs


def euler_step(f, x, t, dt, **params):
    """
    Single step using the Euler Method at x, t.
    
    Parameters
    ----------
    f : function
        The function representing the ODE system to solve.
    x : np.array
        Solution at time t.
    t : float
        The initial time.
    dt : float
        The step size.
    **params : 
        Additional parameters.

    Returns
    ----------
    xn : np.array
        The state (x) of the system after a single Euler step.
    """
    check_ode_inputs(f, x, t, dt) 
    dxdt = f(x, t, **params)
    xn = x + dt * dxdt
    return xn

def RK4_step(f, x, t, dt, **params):
    """
    Single step using the RK4 Method at x, t.
    
    Parameters
    ----------
    f : function
        The function representing the ODE system to solve.
    x : np.array
        Solution at time t.
    t : float
        The initial time.
    dt : float
        The step size.
    **params : 
        Additional parameters.

    Returns
    ----------
    xn : np.array
        The state (x) of the system after a single RK4 step.
    """
    check_ode_inputs(f, x, t, dt) 
    k1 = np.array(f(x, t, **params))
    k2 = np.array(f(x + 0.5*dt*k1, t + 0.5*dt, **params))
    k3 = np.array(f(x + 0.5*dt*k2, t + 0.5*dt, **params))
    k4 = np.array(f(x + dt*k3, t + dt, **params))
    
    xn = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return xn

def heun_step(f, x, t, dt, **params):
    """
    Single step using the Heun Method at x, t.
    
    Parameters
    ----------
    f : function
        The function representing the ODE system to solve.
    x : np.array
        Solution at time t.
    t : float
        The initial time.
    dt : float
        The step size.
    **params : 
        Additional parameters.

    Returns
    ----------
    xn : np.array
        The state (x) of the system after a single Heun step.
    """
    check_ode_inputs(f, x, t, dt)
    k1 = f(x, t, **params)
    k2 = f(x + dt*k1, t + dt, **params)
    xn = x + 0.5 * dt * (k1 + k2)
    return xn

def solve_to(f, X0, t0, t1, dtmax, method, **params):
    """
    Solve an ODE from t0 to t1 using specified method.
    
    Parameters
    ----------
    f : function
        The function representing the ODE system to solve.
    X0 : np.array
        Initial values of solution.
    t0 : float
        The time to solve from.
    t1 : float
        The time to solve to.
    dtmax : float
        The maximum step size.
    method : string
        The method to use, either 'Euler' or 'RK4'.
    **params :
        Additional parameters for the ODE.
    
    Returns
    ----------
    X1 : np.array
        The state of the system after the final time.
    """

    method_dict = {'Euler': euler_step, 'RK4': RK4_step, 'Heun': heun_step}
    method = method_dict.get(method)
    if method is None:
        raise ValueError('Method not found. Please use Euler, RK4 or Heun.')

    X = X0
    t = t0
    dt = dtmax
    while t < t1:
        if t + dt < t1:
            X = method(f, X, t, dt, **params)
            t += dt
        else:
            dt = t1 - t
            X = method(f, X, t, dt, **params)
            t += dt
    return X

def solve_ode(f, X0, t, method, **params):
    """
    Solve an ODE to a set of times using specified method.

    Parameters
    ----------
    f : function
        The function representing the ODE system to solve.
    X0 : np.array
        Initial values of solution.
    t : np.array
        The times to solve to.
    method : string
        The method to use, either 'Euler' or 'RK4'.
    **params :
        Additional parameters for the ODE.

    Returns
    ----------
    X : np.array
        The state of the system at each time in t.
    """
    X0 = np.array(X0)
    if len(X0) > 1:
        X = np.zeros((len(t), X0.shape[0]))
        X[0, :] = X0
    else:
        X = np.zeros((len(t), 1))
        X[0, 0] = X0

    for i in range(len(t) - 1):
        t0 = t[i]
        t1 = t[i + 1]
        X[i + 1] = solve_to(f, X[i], t0, t1, 0.01, method, **params)

    return X



