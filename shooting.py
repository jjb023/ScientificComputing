import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from ODEs import solve_ode
from scipy.optimize import fsolve  


def phase_condition(y_final, y_initial, *params):
    """
    Define a phase condition for the shooting method.
    
    Parameters:
    y_final : array
        The final state from the ODE solver.
    y_initial : array
        The initial state used for solving the ODE.
    params : tuple
        Additional parameters needed for the condition.

    Returns:
    float
        The difference used as part of the root finding process.
    """
    # Example condition: return the difference in one component
    return y_final[0] - y_initial[0] 


def shooting(function, phase_condition):
    """
    Returns function for solving a BVP using shooting method.
    
    Parameters
    ----------
    function : function
        The function representing the ODE system.
    phase_condition : function
        Phase condition for limit cycle.

    Returns
    -------
    function
        The function for solving a BVP using shooting method.
        """
    if not callable(function) or not callable(phase_condition):
        raise ValueError("Both function and phase_cond must be callable functions")
    
    def shooting_func(initial_conditions, params):
        u0, T = initial_conditions[:-1], initial_conditions[-1]
        # Explicitly ensure u0 is an array
        u0 = np.atleast_1d(u0)
        
        sol, t = solve_ode(function, u0, 0, T/1000, T, 'RK4', params=params)
        residuals = np.append(u0 - sol[-1], phase_condition(sol[-1], u0, *params))
        return residuals


    
    return shooting_func

if __name__ == "__main__":
    # Define your ODE system function
    def my_ode_system(t, y, *params):
        # Ensure y is an array, irrespective of input format
        y = np.array([1.0, 2.0])  # Example for a predator-prey model

        if len(y) < 2:
            raise ValueError("The state vector 'y' must have at least two elements.")
        
        a, b = params
        dydt = [-y[0] + a * y[1], -b * y[0] * y[1]]
        return np.array(dydt)
    


    # Initialize parameters and conditions for the shooting function
    a = 0.1
    b = 0.2
    params = (0.1, 0.2)  # Example parameters for the ODE system
    initial_guess = [1.0, 0.0, 2*np.pi]  # Initial guess for the state variables and period

    # Create the shooting function
    shoot_func = shooting(my_ode_system, phase_condition)

    # Solve the BVP using fsolve
    solution = fsolve(shoot_func, initial_guess, args=(params,))
    print("Solution of the BVP:", solution)        
        
        


# def shooting(u0, function, phase_condition, args, method='RK4', tmax=25, dt=0.01):
#     """
#     Function to find limit cycles of specific ODEs using shooting method.
    
#     Parameters
#     ----------
#     u0 : array
#         Initial guess for the solution.
#     function : function
#         The function representing the ODE system.    
#     phase_condition : function
#         Phase condition for limit cycle.
#     args : tuple
#         Additional arguments to pass to the function.
#     method : string
#         The method to use, either 'Euler' or 'RK4'.
#     tmax : float
#         The final time.
#     dt : float
#         The time step size.
        
#     Returns
#     -------
#     u : array
#         The solution to the ODE system that meets the phase condition.
#         """
    
#     # Check if u0 is an array-like object
#     if not isinstance(u0, (list, np.ndarray)):
#         raise ValueError("u0 must be an array-like object")

#     # Check if u0 has at least two elements
#     if len(u0) < 2:
#         raise ValueError("u0 must have at least two elements")

#     def objective(u0):
#         # Solve ODE system
#         u, t = solve_ode(function, u0, 0, dt, tmax, method=method)
#         return phase_condition(u[-1], u0, *args)
    
#     # Find root of objective function
#     result = root(objective, u0, method='hybr')

#     if result.success:
#         u, t = solve_ode(function, result.x, 0, dt, tmax, method)
#         return u, t
#     else:
#         raise ValueError("Root finding didn't converge.")
    

    
    