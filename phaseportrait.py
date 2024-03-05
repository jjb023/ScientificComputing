import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from matplotlib import pyplot
from math import nan

def orbit(ode, initialu, duration):
    sol = solve_ivp(ode, (0, duration), initialu)
    return (sol.t, sol.y)

def nullcline(ode, u0range, index=0, point=101):
    Vval = np.linspace(min(u0range), max(u0range), point)
    Nval = np.zeros(np.size(Vval))
    for (i,V) in enumerate(Vval):
        result = root(lambda N: ode(nan, (V, N))[index], 0)
        if result.success:
            Nval[i] = result.x
        else:
            Nval[i] = nan
    return (Vval, Nval) 

def equilibrium(ode, initialu):
    result = root(lambda u: ode(nan, u), initialu)
    if result.success:
        return result.x
    else:
        return nan
    # TODO: Error message?