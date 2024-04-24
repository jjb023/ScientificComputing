import numpy as np
from scipy.optimize import fsolve
from reportshooting import numshoot
import warnings



def natpar(f, guess, parvalues, parameter, Tguess, phasecondition, discretisation, **params):
    """
    Compute and plot the continuation of a solution of an ODE system with respect to a parameter.

    Parameters:
    f : function
        The ODE system to be solved.
    u0 : array
        Initial guesses.
    pars : tuple
        Containing (minimum parameter value, maximum parameter value, how many values).
    phasecon : function
        The phase condition.
    discretisation : string
        The discretisation method to use, either 'fsolve' or 'shooting'.   
    """
    parmin, parmax, steps = parvalues
    parvals = np.linspace(parmin, parmax, steps)

    Ts = []

    for i in range(steps):
        params[parameter] = parvals[i]
        prevsol = u0[i]

        if discretisation == 'shooting':
            try:
                X0, T = numshoot


    sollist = [] 
    for par in pars:
        try:
            sol = fsolve(f, u0, args=(par,))
        except Exception as e:
            warnings.warn(f"Error encountered in fsolve for par={par}: {e}")
            continue
        sollist.append(sol)
        u0 =sol
    return np.array(sollist), pars

