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
    """
    parmin, parmax, steps = parvalues
    parvals = np.linspace(parmin, parmax, steps)
    # print(parvals) # Debugging

    Ts = [Tguess]
   

    for i in range(steps):
        params[parameter] = parvals[i]
        prevsol = u0[i]
        # print(u0) # Debugging
        # print(prevsol) # Debugging

        if discretisation == 'shooting':
            try:
                # print(u0) # Debugging
                # print(prevsol) # Debugging

                X0, T = numshoot(f, phasecondition, prevsol.copy(), Tguess, **params )
                # print(X0)
                u0.append(list(X0))
                # print(u0) # Debugging
                Ts.append(T)
            except ValueError:
                print(f"Error encountered in numshoot for par={parvals[i]}")
                break
        else:
            r = fsolve(discretisation(f), u0[i], args=params)
            u0.append(r)

        # print(u0)
    #print(u0)    
    u0 = np.array(u0)
    # print(u0)    
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
    
    # print(u0) # Debugging
    x = u0[1:, 0]
    y = u0[1:, 1]
    # print(x) # Debugging
    # print(y) # Debugging   
    plt.plot(parvals, x, label='x')
    plt.plot(parvals, y, label='y')
    plt.xlabel('Parameter')
    plt.ylabel('Solution')
    plt.title('Continuation of solution with respect to parameter')
    plt.legend()
    plt.show()
