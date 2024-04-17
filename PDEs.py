import numpy as np

# Explicit Euler function for Partial Differential Equations
def explicit_euler(u0, dt, dx, D, tmax):
    steps = int(tmax / dt)
    u = u0.copy()
    for _ in range(steps):
        u[1:-1] = u[1:-1] + D * dt / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    return u, np.linspace(0, tmax, steps+1)
    