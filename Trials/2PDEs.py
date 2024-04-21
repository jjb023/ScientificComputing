import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp

def explicit_euler(u0, dt, dx, D, tmax):
    steps = int(tmax / dt)
    u = u0.copy()
    for _ in range(steps):
        u[1:-1] = u[1:-1] + D * dt / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    return u, np.linspace(0, tmax, steps+1)

def method_of_lines(u0, dt, dx, D, tmax):
    def pde(t, u):
        dudt = np.zeros_like(u)
        dudt[1:-1] = D / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
        return dudt
    
    t_span = (0, tmax)
    sol = solve_ivp(pde, t_span, u0, method='RK45', max_step=dt)
    return sol.y, sol.t

def implicit_euler(u0, dt, dx, D, tmax):
    steps = int(tmax / dt)
    n = len(u0)
    k = D * dt / dx**2
    A = diags([-k, 1+2*k, -k], [-1, 0, 1], shape=(n, n)).tocsc()
    u = u0.copy()
    for _ in range(steps):
        u = spsolve(A, u)
    return u, np.linspace(0, tmax, steps+1)

def imex(u0, dt, dx, D, tmax, nonlinear_term):
    steps = int(tmax / dt)
    n = len(u0)
    k = D * dt / dx**2
    A = diags([-k, 1+2*k, -k], [-1, 0, 1], shape=(n, n)).tocsc()
    u = u0.copy()
    for _ in range(steps):
        u_nonlinear = u + dt * nonlinear_term(u)
        u = spsolve(A, u_nonlinear)
    return u, np.linspace(0, tmax, steps+1)

def solve_pde(method, u0, dt, dx, D, tmax, nonlinear_term=None):
    if method == 'explicit_euler':
        return explicit_euler(u0, dt, dx, D, tmax)
    elif method == 'method_of_lines':
        return method_of_lines(u0, dt, dx, D, tmax)
    elif method == 'implicit_euler':
        return implicit_euler(u0, dt, dx, D, tmax)
    elif method == 'imex':
        if nonlinear_term is None:
            raise ValueError("Nonlinear term must be provided for IMEX method")
        return imex(u0, dt, dx, D, tmax, nonlinear_term)
    else:
        raise ValueError("Invalid method specified")

# Example usage
if __name__ == "__main__":
    L = 10
    nx = 100
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    u0 = np.sin(np.pi * x / L)  # Initial condition
    D = 1.0
    dt = 0.01
    tmax = 1.0

    method = 'explicit_euler'  # Can be changed to any of the available methods
    u, t = solve_pde(method, u0, dt, dx, D, tmax)
    plt.plot(x, u[:, -1] if u.ndim > 1 else u, label=f'Solution using {method}')
    plt.legend()
    plt.show()
