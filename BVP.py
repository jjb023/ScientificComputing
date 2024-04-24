import numpy as np


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
