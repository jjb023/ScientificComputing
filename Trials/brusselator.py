from ODEs import solve_ode
import numpy as np
import matplotlib.pyplot as plt

def brusselator_system(x, t, A=1, B=3):
    """
    Compute the time derivative of the Brusselator system.

    Parameters
    ----------
    x : array
        The current state of the system.
    t : float
        The current time.
    A, B : float
        The value of parameter A and B in the Brusselator system.

    Returns
    -------
    dxdt : array
        The time derivative of the Brusselator system.
    """
    x1, x2 = x
    dx1dt = A + x1**2 * x2 - (B+1) * x1
    dx2dt = B * x1 - x1**2 * x2
    return np.array([dx1dt, dx2dt])

# Initial conditions and parameters
x0 = [1, 1]  
t0 = 0     
dt = 0.01  
tmax = 20   

xvals, tvals = solve_ode(brusselator_system, x0, t0, dt, tmax, method='RK4')
# Note: RK4 has higher accuracy than Euler method

# Plot the results
plt.plot(tvals, [x[0] for x in xvals], label='x(t)')
plt.plot(tvals, [x[1] for x in xvals], label='y(t)')
plt.xlabel('Time, t')
plt.ylabel('x and y')
plt.legend()
plt.title('Brusselator System Time Series')
plt.show()
