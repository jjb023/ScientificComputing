import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

#     dxdt = x*(1-x) - (a*x*y) / (d+x)
#     dydt = b*y*(1-(y/x))
# For the predator-prey system, the phase condition is that the gradient of x at t = 0 is 0.
def predator_prey(X, t, params):
    x, y = X
    a, b, d = params
    dxdt = x*(1-x) - (a*x*y) / (d+x)
    dydt = b*y*(1-(y/x))
    X = np.array([dxdt, dydt])
    return X

def phase_condition(X0, **params):
    dxdt = predator_prey(X0, 0, params)
    return dxdt[0]

# Define RK4 method for solving ODEs
def RK4(f, y0, t0, dt, tmax, **kwargs):
    t = np.arange(t0, tmax, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = dt * f(y[i-1], t[i-1], **kwargs)
        k2 = dt * f(y[i-1] + 0.5*k1, t[i-1] + 0.5*dt, **kwargs)
        k3 = dt * f(y[i-1] + 0.5*k2, t[i-1] + 0.5*dt, **kwargs)
        k4 = dt * f(y[i-1] + k3, t[i-1] + dt, **kwargs)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y, t

# Define numerical solver for ODE system
def solve_ode(f, y0, t0, dt, tmax, method, **kwargs):
    if method == 'RK4':
        return RK4(f, y0, t0, dt, tmax, **kwargs)
    else:
        raise ValueError("Invalid method. Choose 'RK4'.")


# Define the shooting method
def shooting(function, phase_condition):
    def shooting_func(initial_conditions, params):
        u0, T = np.atleast_1d(initial_conditions[:-1]), initial_conditions[-1]
        sol, t = solve_ode(function, u0, 0, T/1000, T, 'RK4', params=params)
        residuals = np.append(u0 - sol[-1], phase_condition(sol[-1], u0, *params))
        return residuals
    return shooting_func

# Define a phase condition for the shooting method
def phase_condition(y_final, y_initial, *params):
    return y_final[0] - y_initial[0]

# Main function to setup and solve the boundary value problem
def main():
    params = (1.0, 0.2, 0.1)  # Example parameters: a, b, d
    initial_guess = [0.5, 0.5, 2*np.pi]  # Initial state for x, y, and guessed period
    initial_guess = np.array(initial_guess)  # Optionally convert to numpy array for consistency

    # Create the shooting function using the predator-prey model
    shoot_func = shooting(predator_prey, phase_condition)

    # Solve the BVP using fsolve
    solution = fsolve(shoot_func, initial_guess, args=(params,))
    print("Solution of the BVP:", solution)

    # Plot the solution
    t_eval = np.linspace(0, solution[-1], 1000)
    sol = solve_ode(predator_prey, solution[:-1], 0, solution[-1]/1000, solution[-1], 'RK4', params=params)
    plt.plot(t_eval, sol[0][:,0], label='Prey (x)')
    plt.plot(t_eval, sol[0][:,1], label='Predator (y)')
    plt.title('Predator-Prey System with Limit Cycle')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
