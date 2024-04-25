import numpy as np
from scipy.optimize import fsolve
from shooting import numshoot
import warnings
import matplotlib.pyplot as plt




def natpar(f, u0, parvalues, parameter, Tguess, phasecondition, discretisation, **params):
    """
    Compute and plot the continuation of a solution of an ODE system with respect to a parameter.

    Parameters:
    f : function
        The ODE system to be solved.
    u0 : array
        Initial guesses.
    parsvalues : tuple
        Containing (minimum parameter value, maximum parameter value, how many values).
    phasecon : function
        The phase condition.
    discretisation : string
        The discretisation method to use, either 'fsolve' or 'shooting'.   
    **params : any additional parameters.

    Returns:
    u0 : array
        The solution of the ODE system.
    parvals : array
        The parameter values.
    Ts : array
        The time values.
    """
    parmin, parmax, steps = parvalues
    parvals = np.linspace(parmin, parmax, steps)
    

    Ts = [Tguess]
   

    for i in range(steps):
        params[parameter] = parvals[i]
        prevsol = u0[i]
        
        

        if discretisation == 'shooting':
            try:
                X0, T = numshoot(f, phasecondition, prevsol.copy(), Tguess, **params )
                u0.append(list(X0))
                Ts.append(T)
            except ValueError:
                print(f"Error encountered in numshoot for par={parvals[i]}")
                break
        else:
            r = fsolve(discretisation(f), u0[i], args=params)
            u0.append(r)
    u0 = np.array(u0)
    
    return u0, parvals, Ts

def pseudoarclengthcontinuation(f, u0, parvalues, parameter, Tguess, phasecondition, discretisation, **params):
    """
    Use the pseudo-arclength continuation method to compute and plot the continuation of a solution of an ODE system with respect to a parameter.

    Parameters:
    f : function
        The ODE system to be solved.
    u0 : array
        Initial guesses.
    parsvalues : tuple
        Containing (minimum parameter value, maximum parameter value, how many values).
    phasecon : function
        The phase condition.
    discretisation : string
        The discretisation method to use, either 'fsolve' or 'shooting'.
    **params : any additional parameters.

    Returns:
    u0 : array
        The solution of the ODE system.
    parvals : array
        The parameter values.
    Ts : array
        The time values.
    """
    parmin, parmax, steps = parvalues
    parvals = np.linspace(parmin, parmax, steps)
    

    Ts = [Tguess]
    u0 = [u0]
    du0 = np.zeros_like(u0[0])
    du0[0] = 1
    du0 = du0/np.linalg.norm(du0)
    for i in range(steps):
        params[parameter] = parvals[i]
        prevsol = u0[i]
        
        if discretisation == 'shooting':
            try:
                X0, T = numshoot(f, phasecondition, prevsol.copy(), Tguess, **params )
                u0.append(list(X0))
                Ts.append(T)
            except ValueError:
                print(f"Error encountered in numshoot for par={parvals[i]}")
                break
        else:
            r = fsolve(discretisation(f), u0[i], args=params)
            u0.append(r)
        du0 = du0/np.linalg.norm(du0)
        u0[i+1] = u0[i] + 0.01*du0
    u0 = np.array(u0)
    
    return u0, parvals, Ts


def plot_continuation(u0, parvals):
    """
    Plot the continuation of a solution of an ODE system with respect to a parameter.

    Parameters:
    u0 : array
        The solution of the ODE system.
    parvals : array
        The parameter values.
    """
    
    
    x = u0[1:, 0]
    y = u0[1:, 1]
    
    
    plt.plot(parvals, x, label='x')
    plt.plot(parvals, y, label='y')
    plt.xlabel('Parameter')
    plt.ylabel('Solution')
    plt.title('Continuation of solution with respect to parameter')
    plt.legend()
    plt.show()


