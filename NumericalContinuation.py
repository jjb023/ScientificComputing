import numpy as np
from scipy.optimize import fsolve
from shooting import shooting
import warnings



def natpar(f, u0, parmin, parmax, steps, phasecondition, discretisation, **params):
    """
    Compute and plot the continuation of a solution of an ODE system with respect to a parameter.

    Parameters:
    f : function
        The ODE system.
    u0 : array
        Initial guesses.
    parmin : float
        Minimum parameter value.
    parmax : float
        Maximum parameter value.
    steps : int
        Number of steps.
    phasecon : function
        The phase condition.
    discretisation : string
        The discretisation method to use, either 'fsolve' or 'shooting'.   
    """

    sollist = []
    pars = np.linspace(parmin, parmax, steps)  
    for par in pars:
        try:
            sol = fsolve(f, u0, args=(par,))
        except Exception as e:
            warnings.warn(f"Error encountered in fsolve for par={par}: {e}")
            continue
        sollist.append(sol)
        u0 =sol
    return np.array(sollist), pars

