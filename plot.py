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

def plotsystemode(X, T):
    """
    Plot the solution to an ODE.
    
    Parameters
    ----------
    X : array
        The solution to the ODE.
    T : array
        The time values for the solution.
    """
    plt.plot(T, X[0], label='x(t)', color='blue')
    plt.plot(T, X[1], label='y(t)', color='red')
    plt.title('Time Series of x(t) and y(t)')  # Title of the plot
    plt.xlabel('Time t')  # Label for the x-axis
    plt.ylabel('State Variables x and y')  # Label for the y-axis
    plt.legend()  # Show legend to label the lines
    plt.grid(True)  # Show grid lines for better readability
    plt.show()  # Display the plot  




