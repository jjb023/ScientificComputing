import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the ODE system using a function
def example_ode_system(X, t, params):
    """
    An example ODE system (linear dynamics for demonstration).
    Args:
        X (array): state vector of the system
        t (float): current time
        params (tuple): parameters of the system (a, b)
    Returns:
        array: derivative of the state vector
    """
    a, b = params  # unpack parameters
    x, y = X
    dxdt = -a * x + b * y
    dydt = b * x - a * y
    return np.array([dxdt, dydt])

# Shooting function to find periodic orbit
def shooting_function(initial_conditions, period, ode_system, params):
    """
    Function to be zeroed by fsolve.
    Args:
        initial_conditions (array): guessed initial conditions
        period (float): guessed period
        ode_system (function): ODE system function
        params (tuple): parameters for the ode_system
    Returns:
        array: error in initial conditions and period after one period
    """
    t_eval = np.linspace(0, period, 1000)
    sol = solve_ode(ode_system, initial_conditions, 0, period/1000, period, 'RK4', params)
    error = sol[0][-1] - initial_conditions
    return error

# Main function to setup and solve the boundary value problem
def main():
    initial_conditions = np.array([1.0, 0.0])
    guessed_period = 2 * np.pi
    params = (1.0, 0.3)  # Ensure this is a tuple

    # Use fsolve to find the correct initial conditions and period
    result = fsolve(shooting_function, np.hstack([initial_conditions, guessed_period]), args=(example_ode_system, params))

    fixed_point, period = result[:-1], result[-1]
    print(f"Fixed Point: {fixed_point}, Period: {period}")

    # Solve ODE with the found initial conditions and period
    t_eval = np.linspace(0, period, 1000)
    sol = solve_ode(example_ode_system, fixed_point, 0, period/1000, period, 'RK4', params)

    plt.plot(t_eval, sol[0][:,0], label='State x')
    plt.plot(t_eval, sol[0][:,1], label='State y')
    plt.title('Periodic Orbit Found Using Shooting Method')
    plt.xlabel('Time')
    plt.ylabel('State Variables')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
