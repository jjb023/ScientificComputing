import numpy as np
import matplotlib.pyplot as plt
from examplefuncs import *

def euler_step_system(state, dt, system_derivs):
    """
    Perform a single Euler method step for a system of ODEs.
    :param state: Current state of the system [x, y]
    :param dt: Timestep size
    :param system_derivs: Derivative functions for the system
    :return: Updated state after one Euler step
    """
    x, y = state
    dxdt, dydt = system_derivs(x, y)
    x_new = x + dt * dxdt
    y_new = y + dt * dydt
    return [x_new, y_new]

def rk4_step_system(state, dt, system_derivs):
    """
    Perform a single RK4 method step for a system of ODEs.
    :param state: Current state of the system [x, y]
    :param dt: Timestep size
    :param system_derivs: Derivative functions for the system
    :return: Updated state after one RK4 step
    """
    x, y = state
    k1 = np.array(system_derivs(x, y))
    k2 = np.array(system_derivs(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1]))
    k3 = np.array(system_derivs(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1]))
    k4 = np.array(system_derivs(x + dt * k3[0], y + dt * k3[1]))
    new_state = [x + dt * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6,
                 y + dt * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6]
    return new_state

# Define a function to solve the system from t1 to t2
def solve_system(t1, t2, deltat_max, method='euler'):
    times = np.arange(t1, t2 + deltat_max, deltat_max)
    states = [[1, 0]]  # Initial conditions: x(0)=1, y(0)=0 (y = dx/dt, hence y(0) = x'(0) = 0)

    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        if method == 'euler':
            states.append(euler_step_system(states[-1], dt, system_derivs))
        elif method == 'rk4':
            states.append(rk4_step_system(states[-1], dt, system_derivs))

    return times, np.array(states)

def system_derivs(x, y):
    return [y, -x]  # dx/dt = y, dy/dt = -x

# Solve the system and plot the results
times, states = solve_system(0, 10, 0.1, 'rk4')

plt.figure(figsize=(12, 6))
plt.plot(times, states[:, 0], label='x(t) - Position')
plt.plot(times, states[:, 1], label='y(t) - Velocity (= dx/dt)')
plt.title("System Behavior Over Time Using RK4")
plt.xlabel("Time t")
plt.ylabel("State Variables x and y")
plt.legend()
plt.grid(True)
plt.show()

# Plot x vs y to see the phase space trajectory
plt.figure(figsize=(6, 6))
plt.plot(states[:, 0], states[:, 1])
plt.title("Phase Space Trajectory (x vs y) Using RK4")
plt.xlabel("Position x(t)")
plt.ylabel("Velocity y(t)")
plt.grid(True)
plt.show()
