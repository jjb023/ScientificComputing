import numpy as np


def checkf(f):

    if not callable(f):
        raise TypeError("Input function is not callable")
    
def checkx(x):
    if not isinstance(x, (list, tuple, np.ndarray)):
        raise TypeError("Initial conditions must be a list, tuple, or numpy array")
    

def checkt(t):

    if not isinstance(t, (int, float)):
        raise TypeError("Initial time must be a number")
    
def checkdt(dt):
    
    if not isinstance(dt, (int, float)):
        raise TypeError("Step size must be a number")
    
    if dt <= 0:
        raise ValueError("Step size must be positive")
    
def checkmaxstep(maxstep):
        
        if not isinstance(maxstep, (int, float)):
            raise TypeError("Maximum step size must be a number")
        
        if maxstep <= 0:
            raise ValueError("Maximum step size must be positive")
    
def checkparams(params):
    
    if not isinstance(params, tuple):
        raise TypeError("Additional parameters must be a tuple")
    
def checktmax(tmax):
    
    if not isinstance(tmax, (int, float)):
        raise TypeError("Final time must be a number")
    
    if tmax <= 0:
        raise ValueError("Final time must be positive")
    
def checkmethod(method):
    
    if method not in ["Euler", "RK4"]:
        raise ValueError("Method must be 'Euler' or 'RK4'")
    

def checkteval(teval):
    if not isinstance(teval, (list, tuple, np.ndarray)):
        raise TypeError("Evaluation times must be a list or tuple")
    for t in teval:
        if not isinstance(t, (int, float)):
            raise TypeError("Each evaluation time must be a number")
        


    
