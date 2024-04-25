import numpy as np
import matplotlib.pyplot as plt
from odes import solve_ode
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp



# Brusselator ODE

def brusselator(X, t, **params):
    x, y = X
    A = params['A']
    B = params['B']

    dxdt = A + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y

    X = [dxdt, dydt]
    return X

def brusspc(X0, **params):
    """
    Phase condition for the Brusselator ODE.
    
    Parameters
    ----------
    X0 : array
        The initial state of the system.
    params : dict
        The parameters of the Brusselator system (A, B).
        
    Returns
    ----------
    dxdt_0 : float
        The gradient at t = 0.
    """
    
    pcbruss = brusselator(X0, 0, **params)[0]
    
    return pcbruss

def hopf(X, t, **params):
    x, y, z = X
    mu = params['mu']
    dxdt = mu * x - y - z + x * (x**2 + y**2 + z**2) - x * (x**2 + y**2 + z**2)**2
    dydt = x + mu * y - z + y * (x**2 + y**2 + z**2) - y * (x**2 + y**2 + z**2)**2
    dzdt = x + y + mu * z + z * (x**2 + y**2 + z**2) - z * (x**2 + y**2 + z**2)**2
    X = [dxdt, dydt, dzdt]
    return X

def hopfpc(X0, **params):
    pchopf = hopf(X0, 0, **params)[0]
    return pchopf

def findroot(X0, *data):
    """
    Function to find the root of the system.

    Parameters
    ----------
    X0 : array
        Initial conditions.
    data : tuple
        Tuple containing the ODE system, phase condition and parameters.

    Returns
    ----------
    output : array
        The difference between the initial conditions and the phase condition.
    """

    T = X0[-1]
    X0 = X0[:-1]
    t = np.linspace(0, T, 100)

    f, phase_condition, params = data if len(data) == 3 else data + (None,)
    if params is not None:
        solution = solve_ode(f, X0, t, 'RK4', **params)
    else:
        solution = solve_ode(f, X0, t, 'RK4')

    if params is not None:
        output = np.append(X0 - solution[-1, :], phase_condition(X0, **params))
    else:
        output = np.append(X0 - solution[-1, :], phase_condition(X0))

    return output


def numshoot(f, phasecondition, X0, Tguess, **params):
    """
    Returns the initial conditions and period of a periodic orbit in an ODE
    system using the numerical shooting method.

    Parameters
    ----------
    f : function 
        The ODE system.
    phasecondition : function
        Function that takes the initial conditions and returns the values of some specific variable(s) at the end of the time interval.
    X0 : list
        Initial conditions.
    Tguess : float
        Initial guess for the period of the periodic orbit.
    params : any parameters.

    Returns
    ----------
    X0 : list
        The initial conditions of the periodic orbit.
    T : float
        The period of the periodic orbit.
    """

    X0T = X0.copy()
    # print(X0_with_T)  ## Debugging
    X0T.append(Tguess)
    # print(X0_with_T)  ## Debugging

    data = (f, phasecondition, params) if params else (f, phasecondition)

    sol = fsolve(findroot, X0T, args=data)

    if sol[:-1].all() == np.array(X0).all() and sol[-1] == Tguess:
        print('Root not found, returning empty array.')
        return []
    
    X0 = sol[:-1]
    T = sol[-1]

    return X0, T

def plotorbit(f, X0, T, **params):

    t = np.linspace(0, T, 1000)
    X = solve_ode(f, X0, t, 'RK4', **params)
    nvars = X.shape[1]
    labels = ['x(t)', 'y(t)', 'z(t)', 'w(t)', 'v(t)']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i in range (nvars):
        plt.plot(t, X[:, i], label = labels[i], color = colors[i])
    plt.xlabel('t')
    plt.ylabel('State Variables')
    plt.title("title")
    plt.legend()
    plt.show()



