import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.optimize import root

def euler_step_system(state, dt, system_derivs, time, params):
    """
    Perform a single Euler method step for a system of ODEs including additional parameters.

    Parameters:
    - state: Current state of the system
    - dt: Timestep size
    - system_derivs: Derivative functions for the system
    - time: Current time
    - params: Additional parameters required by the system

    Returns:
    - Updated state after one Euler step
    """
    derivs = system_derivs(state, time, params)
    new_state = [state[i] + dt * derivs[i] for i in range(len(state))]
    return new_state

def rk4_step_system(state, dt, system_derivs, time, params):
    """
    Perform a single RK4 method step for a system of ODEs including additional parameters.

    Parameters:
    - state: Current state of the system
    - dt: Timestep size
    - system_derivs: Derivative functions for the system
    - time: Current time
    - params: Additional parameters required by the system

    Returns:
    - Updated state after one RK4 step
    """
    k1 = np.array(system_derivs(state, time, params))
    k2 = np.array(system_derivs([state[i] + 0.5 * dt * k1[i] for i in range(len(state))], time + 0.5 * dt, params))
    k3 = np.array(system_derivs([state[i] + 0.5 * dt * k2[i] for i in range(len(state))], time + 0.5 * dt, params))
    k4 = np.array(system_derivs([state[i] + dt * k3[i] for i in range(len(state))], time + dt, params))
    new_state = [state[i] + dt * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6 for i in range(len(state))]
    return new_state

def solve_system(ode_system, initial_conditions, t_range, dt, params, method='euler'):
    """
    Solves any system of ODEs from t_range[0] to t_range[1] using the specified numerical method.
    
    Parameters:
    - ode_system: Function defining the system of ODEs
    - initial_conditions: List of initial values for the system variables
    - t_range: Tuple of (start_time, end_time)
    - dt: Time step for the numerical method
    - params: Parameters required by the ode_system
    - method: Numerical method to use ('euler' or 'rk4')
    
    Returns:
    - times: Array of time points
    - states: Array of system states at each time point
    """
    times = np.arange(t_range[0], t_range[1] + dt, dt)
    states = [initial_conditions]

    for i in range(1, len(times)):
        current_time = times[i-1]
        current_state = states[-1]
        if method == 'euler':
            next_state = euler_step_system(current_state, dt, ode_system, current_time, params)
        elif method == 'rk4':
            next_state = rk4_step_system(current_state, dt, ode_system, current_time, params)
        states.append(next_state)

    return times, np.array(states)

# Example usage with the predator-prey model defined earlier
def predator_prey(X, t, params):
    x, y = X
    a, b, d = params
    dxdt = x * (1 - x) - (a * x * y) / (d + x+1e-5)
    dydt = b * y * (1 - (y / x + 1e-5))
    return np.array([dxdt, dydt])


# Parameters for the predator-prey model
params = (1, 0.3, 0.1)  # Example: (a, b, c, d)
initial_conditions = [0.5, 0.2]  # Example: [prey, predator]
t_range = (0, 100)  # Example: (start_time, end_time)


# Solving the system
times, states = solve_system(predator_prey, initial_conditions, t_range, 0.01, params, 'rk4')

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(times, states[:, 0], label='Prey Population')
plt.plot(times, states[:, 1], label='Predator Population')
plt.title("Predator-Prey Dynamics Over Time Using RK4")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()


# Plotting phase plane
plt.figure(figsize=(8, 6))
plt.plot(states[:, 0], states[:, 1])
plt.title("Phase Plane Diagram")
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
plt.grid(True)
plt.show()

# Analyzing time series to find the period
prey_population = states[:, 0]
peaks, _ = find_peaks(prey_population, height=np.mean(prey_population))
periods = np.diff(times[peaks])  # Time between peaks
average_period = np.mean(periods)

print(f"Estimated Period: {average_period} time units")
print(f"Starting Conditions for Periodic Orbit: Prey = {initial_conditions[0]}, Predators = {initial_conditions[1]}")



def phase_condition(initial_conditions, params, period, system_derivs, initial_guess):
    """
    Compute the phase condition for the limit cycle. Now returns an array of two elements.
    - initial_conditions: initial state of the system
    - params: parameters for the ODE system
    - period: estimated period of the cycle
    - system_derivs: function computing the derivatives of the system
    - initial_guess: the initial guess for the fsolve to ensure the second returned value
    """
    # Solve the system over one period
    times, states = solve_system(system_derivs, initial_conditions, (0, period), 0.01, params, 'rk4')
    final_state = states[-1]
    initial_velocity = system_derivs(initial_conditions, 0, params)
    
    # Orthogonality condition
    orthogonality = np.dot(final_state - initial_conditions, initial_velocity)
    
    # Additional dummy condition: Let's keep the prey population close to its initial guess
    prey_condition = initial_conditions[0] - initial_guess[0]

    return np.array([orthogonality, prey_condition])






