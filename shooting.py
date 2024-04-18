import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from ODEs import solve_ode
from scipy.optimize import fsolve  
from examplefuncs import phase_condition, predator_prey


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
        u0, T = np.atleast_1d(initial_conditions[:-1], initial_conditions[-1])
        print(f"Shooting func - u0: {u0}, T: {T}, params: {params}")
        sol, t = solve_ode(function, u0, 0, T/1000, T, 'RK4', params=params)
        residuals = np.append(u0 - sol[-1], phase_condition(sol[-1], u0, *params))
        return residuals


    return shooting_func

if __name__ == "__main__":
    params = (1.0, 0.2, 0.1)  # Example parameters: a, b, d
    initial_guess = [0.5, 0.5, 2*np.pi]  # Initial state for x, y, and guessed period
    initial_guess = np.array(initial_guess)  # Optionally convert to numpy array for consistency


    # Create the shooting function using the predator-prey model
    shoot_func = shooting(predator_prey, phase_condition)

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
    

    
    