


def checkf(f):

    if not callable(f):
        raise TypeError("Input function is not callable")
    
def checkx0(x0):
    
    if not isinstance(x0, (int, float)):
        raise TypeError("Initial state must be a number")

def checkt0(t0):

    if not isinstance(t0, (int, float)):
        raise TypeError("Initial time must be a number")
    
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