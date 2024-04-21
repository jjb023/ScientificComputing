import numpy as np
import matplotlib.pyplot as plt
from ODEs import solve_ode  # Make sure to replace 'your_module_name' with the actual name of your Python file without the .py extension

def analytical_solution(t):
    """ Analytical solution of dx/dt = -x """
    return np.exp(-t)

def test_ode_solver():
    # Define the ODE dx/dt = -x
    def simple_ode(x, t):
        return -x

    # Test parameters
    x0 = 1.0  # initial condition
    t0 = 0.0  # start time
    tmax = 10.0  # end time
    dt = 0.01  # time step

    # Numerical solutions
    euler_x, euler_t = solve_ode(simple_ode, x0, t0, dt, tmax, 'Euler')
    rk4_x, rk4_t = solve_ode(simple_ode, x0, t0, dt, tmax, 'RK4')

    # Analytical solution for comparison
    t_exact = np.linspace(t0, tmax, num=int(tmax/dt)+1)
    x_exact = analytical_solution(t_exact)

    # Calculate errors
    euler_error = np.abs(euler_x - analytical_solution(np.array(euler_t)))
    rk4_error = np.abs(rk4_x - analytical_solution(np.array(rk4_t)))

    # Plot results
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(euler_t, euler_x, 'b--', label='Euler Method')
    plt.plot(rk4_t, rk4_x, 'r-', label='RK4 Method')
    plt.plot(t_exact, x_exact, 'g', label='Analytical Solution')
    plt.title('ODE Solution Comparison')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(euler_t, euler_error, 'b--', label='Euler Error')
    plt.plot(rk4_t, rk4_error, 'r-', label='RK4 Error')
    plt.title('Error Analysis')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Test with incorrect inputs
    try:
        _, _ = solve_ode(simple_ode, 'incorrect input', t0, dt, tmax, 'Euler')
    except Exception as e:
        print(f"Handled an error with incorrect inputs: {e}")

    try:
        _, _ = solve_ode(simple_ode, x0, t0, dt, tmax, 'Unknown Method')
    except Exception as e:
        print(f"Handled an error with incorrect method: {e}")

if __name__ == "__main__":
    test_ode_solver()
