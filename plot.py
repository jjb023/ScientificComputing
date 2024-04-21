import numpy as np
from finalODEs import solve_ode
import matplotlib.pyplot as plt

def plotode(X, T):
    """
    Plot the solution to an ODE.
    
    Parameters
    ----------
    X : array
        The solution to the ODE.
    T : array
        The time values for the solution.
    """
    plt.plot(T, X)
    plt.show()

