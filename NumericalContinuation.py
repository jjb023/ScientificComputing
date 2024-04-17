import numpy as np
from scipy.optimize import fsolve
from shooting import shooting


def natural_parameter_continuation(f, df, parameter_range, initial_guess, discretisation, step_size=0.01):
    """
    Natural parameter continuation method.
    
    :param f: Function of the variable and parameter, f(x, p).
    :param df: Partial derivative of f with respect to the variable x.
    :param parameter_range: Tuple containing the start and end values of the parameter.
    :param initial_guess: Initial guess for the root at the start of the parameter range.
    :param discretisation: Discretisation method to use.
    :param step_size: Step size for increasing the parameter.
    :return: Array of parameter values and their corresponding roots.
    """
    p_values = np.arange(parameter_range[0], parameter_range[1] + step_size, step_size)
    roots = np.empty_like(p_values)
    x = initial_guess

    for i, p in enumerate(p_values):
        if discretisation == 'shooting':
            try:

        func = lambda x: f(x, p)
        dfunc = lambda x: df(x, p)
        x = discretisation(func, dfunc, x)
        roots[i] = x

    return p_values, roots
    

# Example Usage
if __name__ == "__main__":
    # Define your function and its derivative here
    def example_func(x, p):
        return np.cos(x) - p
    
    def example_dfunc(x, p):
        return -np.sin(x)
    
    # Set the range for the parameter 'p' from 0 to 2
    p_range = (0, 2)
    # Initial guess for the root of 'example_func' when p=0
    initial_x = 1.0
    
    # Perform natural parameter continuation
    parameters, solutions = natural_parameter_continuation(
        example_func, example_dfunc, p_range, initial_x
    )
    
    print("Parameter values:", parameters)
    print("Corresponding roots:", solutions)
