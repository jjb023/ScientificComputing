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
    # plt.plot(X[:, 0], X[:, 1])
    # plt.title('Phase Portrait of state variables')  # Title of the plot
    # plt.xlabel('State Variables')  # Label for the x-axis
    # plt.ylabel('State Variables')  # Label for the y-axis
    # plt.legend()  # Show legend to label the lines
    # plt.grid(True)  # Show grid lines for better readability
    # plt.show()  # Display the plot
    n_vars = X.shape[1] 
    if n_vars == 2:
        plt.figure(figsize=(8, 6))
        plt.plot(X[:, 0], X[:, 1], label='Trajectory')
        plt.title('2D Phase Portrait')
        plt.xlabel('x(t)')
        plt.ylabel('y(t)')
        plt.legend()
        plt.grid(True)
        plt.show()
    elif n_vars == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[:, 0], X[:, 1], X[:, 2], label='Trajectory')
        ax.set_title('3D Phase Portrait')
        ax.set_xlabel('x(t)')
        ax.set_ylabel('y(t)')
        ax.set_zlabel('z(t)')
        ax.legend()
        ax.grid(True)
        plt.show()
    else:
        print(f"Phase portrait not supported for {n_vars} state variables.")




