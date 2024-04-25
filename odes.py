import numpy as np

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
        
    dxdt = f(x, t, **params)
    xn = x + dt * np.array(dxdt)

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
    # print(x) # Debugging
    k1 = np.array(f(x, t, **params))
    k2 = np.array(f(x + 0.5*dt*k1, t + 0.5*dt, **params))
    k3 = np.array(f(x + 0.5*dt*k2, t + 0.5*dt, **params))
    k4 = np.array(f(x + dt*k3, t + dt, **params))
    
    xn = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return xn

# Solvers

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
    # Create a dictionary for the method
    method_dict = {'Euler': euler_step, 'RK4': RK4_step}
    method = method_dict[method]

    X = X0
    t = t0
    dt = dtmax
    while t < t1:
        if t + dt > t1:
            X = method(f, X, t, dt, **params)
            t += dt
        else:
            dt = t1 - t
            X = method(f, X, t, dt, **params)
            t += dt
    return X

def solve_ode(f, X0, t, method, **params):
    """
    Solve an ODE to 
    """
    X0 = np.array(X0)
    if len(X0) > 1:
        X = np.zeros((len(t), X0.shape[0]))
        X[0, :] = X0
    else:
        X = np.zeros(len(t))
        X[0] = X0

    for i in range(len(t) - 1):
        t0 = t[i]
        t1 = t[i + 1]
        X[i + 1] = solve_to(f, X[i], t0, t1, 0.01, method, **params)

    return X



