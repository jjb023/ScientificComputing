import numpy as np
from BVP import Nmatrix, dirichlet
import matplotlib.pyplot as plt


def xgrid(N, a, b):
    """
    Creates a grid of N points between a and b.
    
    Parameters
    ----------
    N : int
        Number of grid points.
    a : float
        Lower bound of x.
    b : float
        Upper bound of x.
        
    Returns
    ----------
    x : np.array
        Grid points.
    dx : float
        Grid spacing.
    xint : np.array
        x interval.
    
    """
    x = np.linspace(a, b, N+1)
    dx = (b - a) / N
    xint = x[1:-1]
    return x, dx, xint



def maxdt(D, dx):
    """
    Calculates the maximum time step size for stability.
    
    Parameters
    ----------
    D : float
        Diffusion coefficient.
    dx : float
        Grid spacing.
        
    Returns
    ----------
    dt : float
        Maximum time step size.
    """

    dt = 0.5 * (dx**2 / D)
    return dt


def explicit_euler(N, D, alpha, beta, a, b, dt, dx,  T, xint, IC):
    """
    Solves the 1D diffusion equation using the explicit Euler method.
    
    Parameters
    ----------
    N : int
        Number of grid points.
    D : float
        Diffusion coefficient.
    alpha : float
        Boundary condition at x=left boundary.
    beta : float
        Boundary condition at x=right boundary.
    a : float
        Lower bound of x.
    b : float
        Upper bound of x.
    dt : float
        Time step size.
    dx : float
        Grid spacing.
    t : float
        Initial time.
    nt : int
        Number of time steps.
    xint : array
        x interval.
    IC : string
        Type of initial condition.
        
    Returns
    ----------
    u : np.array
        Solution to the 1D diffusion equation.
    """
    # Initialize the solution
    C = (D * dt)/(dx**2)
    nt = int(np.ceil(T/dt))
    u = np.zeros((nt+1, N-1))
    u[0, :] = IC(xint, b)
    t = np.linspace(0, T, nt)


   #  Matrices
    A = Nmatrix(N, D)
    B = A @ dirichlet(N, alpha, beta)

    # Time-stepping loop
    for n in range(0, nt):
        u[n+1, :] = u[n, :] + C * (A @ u[n, :] + B)

    return u, t

def implicit_euler(N, D, alpha, beta, a, b, dt, dx, T, xint, IC):
    """
    Solves the 1D diffusion equation using the implicit Euler method.
    
    Parameters
    ----------
    N : int
        Number of grid points.
    D : float
        Diffusion coefficient.
    alpha : float
        Boundary condition at x=left boundary.
    beta : float
        Boundary condition at x=right boundary.
    a : float
        Lower bound of x.
    b : float
        Upper bound of x.
    dt : float
        Time step size.
    dx : float
        Grid spacing.
    t : float
        Initial time.
    nt : int
        Number of time steps.
    xint : array
        x interval.
    IC : string
        Type of initial condition.
        
    Returns
    ----------
    u : np.array
        Solution to the 1D diffusion equation.
    """
    # Initialize the solution
    C = (D * dt)/(dx**2)
    nt = int(np.ceil(T/dt))
    u = np.zeros((nt+1, N-1))
    u[0, :] = IC(xint, b)
    t = np.linspace(0, T, nt)

    # Matrices
    A = Nmatrix(N, D)
    B = A @ dirichlet(N, alpha, beta)

    # Time-stepping loop
    for n in range(0, nt-1):
        u[n+1, :] = np.linalg.solve(np.eye(N-1) - C*A, u[n, :] + C*B) 

    return u, t




def diffusionIC(xint, b):
    """
    Initial condition for the 1D diffusion equation.
    u(x, 0) = 0.5 * x * (L-x)
    
    Parameters
    ----------
    xint : array
        x interval.
    a : float
        Lower bound of x.
    b : float
        Upper bound of x.
        
    Returns
    ----------
    IC : np.array
        Initial condition same shape as xint.
    """
    IC = 0.5 * xint * (b - xint)
    return IC

def convection_matrix(N, P, dx, scheme='upwind'):

    C = np.zeros((N, N))


    if scheme == 'upwind':
        for i in range(1, N-1):  
            C[i, i] = -P / dx
            C[i, i-1] = P / dx
    elif scheme == 'central':
        for i in range(1, N-1):  
            C[i, i+1] = P / (2 * dx)
            C[i, i-1] = -P / (2 * dx)

    C[0, 0] = 1
    C[0, 1] = 0

    C[-1, -1] = 1
    C[-1, -2] = 0

    return C


def solve_reaction_convection_diffusion(N, P):
    x, dx, xint = xgrid(N, 0, 1)  
    D = P
    dt = maxdt(D, dx) 
    IC = np.zeros_like(xint) 
    T = 1

    A = Nmatrix(N, D) + convection_matrix(N, P, dx)

    u, t = explicit_euler(N, D, 0, 0.5, 0, 1, dt, dx, T, xint, IC)
    return u, t, xint


def plot_pde(expu, impu, expt, impt):
    """
    Plot the solution to the 1D diffusion equation using the explicit Euler method.
    
    Parameters
    ----------
    u : np.array
        Solution to the 1D diffusion equation.
    t : np.array
        Time points.
    """
       
    plt.plot(expt, expu, 'o', markersize=2, label='Explicit Euler')
    plt.plot(impt, impu, 'o', markersize=1, label='Implicit Euler')
    plt.xlabel('Time')
    plt.ylabel('u')
    plt.legend()
    plt.show()
    




    
    
    

    