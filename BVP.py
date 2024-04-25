import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import timeit


def Nmatrix(N, D):
    """
    Builds N-1xN-1 matrix for finite difference approximation.
    
    Parameters
    ----------
    N : int
        Number of grid points.
    D : float
        Diffusion coefficient.
        
    Returns
    ----------
    A : np.array
        N-1xN-1 matrix.
    """
    
    A = np.zeros((N-1, N-1))
    for i in range(N-1):
        A[i, i] = -2
        if i != 0:
            A[i, i-1] = 1
        if i != N-2:
            A[i, i+1] = 1
    A = D * A
    return A


def dirichlet(N, u0, uN):
    """
    Builds vector representing boundary conditions for finite difference approximation.
    
    Parameters
    ----------
    N : int
        Number of grid points.
    u0 : float
        Boundary condition at x=left boundary.
    uN : float
        Boundary condition at x=right boundary.
        
    Returns
    ----------
    b : np.array
        Vector representing boundary conditions.
    """
    
    b = np.zeros((N-1),)
    b[0] = u0
    b[-1] = uN
    return b

def neumann(N, bc_left, bc_right, dx):
    """
    Builds N-1xN-1 matrix for finite difference approximation with Neumann boundary conditions.
    
    Parameters
    ----------
    N : int
        Number of grid points.
    bc_left : float
        Boundary condition at x=left boundary.
    uN : float
        Boundary condition at x=right boundary.
    dx : float
        Grid spacing.
        
    Returns
    ----------
    A : np.array
        N-1xN-1 matrix.
    """
    
    A = np.zeros((N-1, N-1))
    for i in range(N-1):
        A[i, i] = -2
        if i != 0:
            A[i, i-1] = 1
        if i != N-2:
            A[i, i+1] = 1
    A[0, 0] = -1 - bc_left/dx
    A[-1, -1] = -1 + bc_right/dx
    return A

def solve_poisson(lower_x, upper_x, num_points, D, sigma, bc_left, bc_right, method='dense'):
    """
    Solve the Poisson equation using a finite difference method.
    
    Parameters
    ----------
    lower_x : float
        Lower bound of the domain.
    upper_x : float
        Upper bound of the domain.
    num_points : int
        Number of grid points.
    D : float
        Diffusion coefficient.
    sigma : float
        sigma
    bc_left : float
        Boundary condition at x=left boundary.
    bc_right : float
        Boundary condition at x=right boundary.
    method : string
        Method to use, either 'dense' or 'sparse'.

    Returns
    ----------
    x : np.array
        Grid points.
    u : np.array
        Solution to the Poisson equation.
        """
    x = np.linspace(lower_x, upper_x, num_points)
    dx = x[1] - x[0]  # Grid spacing
    
   
    if method == 'dense':
        A = np.zeros((num_points, num_points))
        np.fill_diagonal(A, -2 * D / dx**2)
        np.fill_diagonal(A[1:], D / dx**2)
        np.fill_diagonal(A[:, 1:], D / dx**2)
    elif method == 'sparse':
        main_diag = np.full(num_points, -2 * D / dx**2)
        off_diags = np.full(num_points - 1, D / dx**2)
        A = sp.diags([off_diags, main_diag, off_diags], offsets=[-1, 0, 1], shape=(num_points, num_points), format='csr')
    
    
   
    A[0, 0] = A[-1, -1] = 1
    A[0, 1] = A[-1, -2] = 0


    b = (1 / (2 * np.pi * sigma**2)) * np.exp(-x**2 / (2 * sigma**2))
    b[0] = bc_left
    b[-1] = bc_right  

    start_time = timeit.default_timer()
    if method == 'dense':
        u = np.linalg.solve(A, b)
    elif method == 'sparse':
        u = spla.spsolve(sp.csr_matrix(A), b)
    elapsed_time = timeit.default_timer() - start_time

    
    index_center = num_points // 2  # Index for x=0 if domain is symmetrical around 0
    print(f"Value of u at u(0) (for sigma={sigma}) :{u[index_center]:.4f}")   
    print(f"Time to solve using {method} method: {elapsed_time:.4f} seconds")

    return x, u, elapsed_time

def plot_poisson(xlist, ulist, labels, title):
    plt.figure(figsize=(8, 4))
    for x, u, label in zip(xlist, ulist, labels):
        plt.plot(x, u, label=label)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


def debug_matrix_vector(A, b, method):
    print(f"Method: {method}")
    print("Matrix A (sample):")
    print(A[:5, :5])  
    print("Vector b (sample):")
    print(b[:5])


