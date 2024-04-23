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

def plotsystemode(X, t):
    """
    Plot the solution to an ODE.
    
    Parameters
    ----------
    X : array
        The solution to the ODE.
    T : array
        The time values for the solution.
    """
    print(X)

    nvars = X.shape[1]
    labels = ['x(t)', 'y(t)', 'z(t)', 'w(t)', 'v(t)']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i in range(nvars):
        plt.plot(t, X[:,i], label=labels[i], color=colors[i])
    plt.title('Time Series of state variables')  # Title of the plot
    plt.xlabel('Time t')  # Label for the x-axis
    plt.ylabel('State Variables')  # Label for the y-axis
    plt.legend()  # Show legend to label the lines
    plt.grid(True)  # Show grid lines for better readability
    plt.show()  # Display the plot  

def plotphaseportrait(X):
    """
    Plot the phase portrait of an ODE.
    
    Parameters
    ----------
    X : array
        The solution to the ODE.
    """
    labels = ['x(t)', 'y(t)', 'z(t)', 'w(t)', 'v(t)']
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    plt.plot(X[:, 0], X[:, 1])
    plt.title('Phase Portrait of state variables')  # Title of the plot
    plt.xlabel('State Variables')  # Label for the x-axis
    plt.ylabel('State Variables')  # Label for the y-axis
    plt.legend()  # Show legend to label the lines
    plt.grid(True)  # Show grid lines for better readability
    plt.show()  # Display the plot




