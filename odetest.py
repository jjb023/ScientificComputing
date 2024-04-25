import numpy as np
import matplotlib.pyplot as plt
import odes

# Define the ODE to be solved
def f(x, t):
    return -2.5 * x

# Initial condition
X0 = np.array([1])

# Time array
t = np.linspace(0, 1, 100)

# Test the euler_step function
x_euler = odes.euler_step(f, X0, 0, 0.01)
print(f"euler_step output: {x_euler}")

# Test the RK4_step function
x_RK4 = odes.RK4_step(f, X0, 0, 0.01)
print(f"RK4_step output: {x_RK4}")

# Test the heun_step function
x_heun = odes.heun_step(f, X0, 0, 0.01)
print(f"heun_step output: {x_heun}")

# Test the solve_to function
x_solve_to = odes.solve_to(f, X0, 0, 1, 0.01, 'Euler')
print(f"solve_to output: {x_solve_to}")

# Test the solve_ode function
X_solve_ode = odes.solve_ode(f, X0, t, 'Euler')

# Plot the numerical solution from solve_ode
plt.plot(t, X_solve_ode, label='Numerical solution')

# Plot the analytical solution
X_analytical = X0[0] * np.exp(-2.5 * t)
plt.plot(t, X_analytical, label='Analytical solution')

plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()