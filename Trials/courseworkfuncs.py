import numpy as np

def brusselator(X, t, params):
    """
    Function to compute the derivatives for the Brusselator system.

    Parameters
    ----------
    X : array
        State vector [x, y].
    t : float
        The evaluation time (not used in this function but included for compatibility).
    params : tuple
        The parameters of the Brusselator system (A, B).

    """

    x, y = X
    A, B = params

    dxdt = A + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y

    dXdt = np.array[dxdt, dydt]
    return dXdt
