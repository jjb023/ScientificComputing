import numpy as np
from BVP import Nmatrix, dirichlet

def grid(N, xlower, xupper):
    """
    Creates a grid of N points between xlower and xupper.
    
    Parameters
    ----------
    N : int
        Number of grid points.
    xlower : float
        Lower bound of x.
    xupper : float
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
    x = np.linspace(xlower, xupper, N+1)
    dx = (xupper - xlower) / N
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
    dt = dx**2 / (2*D)
    return dt





# Explicit Euler function for Partial Differential Equations
def explicit_euler(N, D, bcleft, bcright, xlower, xupper, dt, dx, t, nt, xint, IC):
    """
    Solves the 1D diffusion equation using the explicit Euler method.
    
    Parameters
    ----------
    N : int
        Number of grid points.
    D : float
        Diffusion coefficient.
    bcleft : float
        Boundary condition at x=left boundary.
    bcright : float
        Boundary condition at x=right boundary.
    xlower : float
        Lower bound of x.
    xupper : float
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
    u = np.zeros((nt+1, N-1))
    u[0, :] = IC(xint, xlower, xupper)

    # Matrices
    A = Nmatrix(N, D)
    B = A @ dirichlet(N, bcleft, bcright)

    # Time-stepping loop
    for n in range(0, nt-1):
        u[n+1, :] = u[n, :] + (dt*D/dx**2) * (A @ u[n, :] + B)

    return u




    
    
    

    