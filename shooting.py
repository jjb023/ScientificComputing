import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from finalODEs import solve_ode
from scipy.optimize import fsolve  
from examplefuncs import predator_prey
import matplotlib.pyplot as plt

def findroot(X0, *data):
    T = X0[-1]
    X = X0[:-1]
    t = np.linspace(0, T, 100)
    f, phase_condition, params = data if len(data) == 3 else data + (None,)
    if params is not None:
        solution = solve_ode('rk4', f, t, X0, **params)
    else:
        solution = solve_ode('rk4', f, t, X0)

    if params is not None:
        output = np.append(X0 - solution[-1, :], phase_condition(X0, **params))
    else:
        output = np.append(X0 - solution[-1, :], phase_condition(X0))

    return output

def numerical_shooting(f, phase_condition, X0, T_guess,  **params):
    """
   Returns the initial conditions and period of a periodic orbit in an ODE 
   system using the numerical shooting method.
   
   Parameters: 
   f : (function) The ODE system.
   phase_condition : (function) function that takes the initial conditions 
   and returns the values of some specific variable(s) at the end of the time 
   interval.
   X0 : (list) initial conditions.
   T_guess : (float)  initial guess for the period of the periodic orbit.
   params : any parameters.
       
   """
    X0_with_T = X0.copy()
    X0_with_T.append(T_guess)

    data = (f, phase_condition, params) if params else (f, phase_condition)

    sol = fsolve(findroot, X0_with_T, args=data)

    if sol[:-1].all() == np.array(X0).all() and sol[-1] == T_guess:
        print('Root Finder Failed, returning empty array...')
        return []

    X0 = sol[:-1]
    T = sol[-1]

    return X0, T


def phase_condition(f, u0, *params):
    """
    Returns the phase condition for a ode system.
    
    Parameters
    ----------
    f : function
        The function representing the ODE system.
    u0 : array
        Initial conditions (x0 and t values).
    *params : array
        Additional parameters for ODE.

    Returns
    -------
    x0 : float
        The new initial state of the system.
    t : float
        The new time.
    pc : function
        The phase condition for the ode system.
    """

    x0, t = u0[:-1], u0[-1]
    pc = f(x0, t, *params)[0]

    return x0, t, pc

    


def shooting(f):
    """
    Returns function to be solved using shooting method.
    
    Parameters
    ----------
    function : function
        The function representing the ODE system.
        
    Returns
    -------
    function
        The function to be solved using shooting method.
    """
    def residuals(u0, phasecondition, *params):
        """
        Find and set up conditions that are solved for.
        
        Parameters
        ----------
        u0 : array
            Initial conditions (x0 and t values).
        phasecondition : function
            Phase condition for limit cycle.
        *params : array
            Additional parameters for ODE.
        
        Returns
        -------
        residuals : array
            Residuals of the initial conditions.
        """
        x0, t, pc = phasecondition(f, u0, *params)

        solx, solt = solve_ode(f, x0, 0, t, 0.01, 'RK4', True, params=params)

        period = []

        for i in range(len(x0)):
            period.append(x0[i] - solx[-1][i])

        full_residuals = np.append(period, pc)

        return full_residuals
    
    return residuals

def plotsol(ode, phasecondition, u0, args=(), dtmax=0.01):

    def shooting2(ode, u0, phasecondition, method, **params):
        G = shooting(ode)
        orbit = method(G, u0, phasecondition, **params)
        print(orbit)
        return orbit
    
    def shootingG(ode):
        def G(x0, phasecondition, **params):
            u0, t = x0[:-1], x0[-1]
            g = np.append(u0 - solve_ivp(ode, [0, t], u0, args=params, method='RK45', t_eval=[t]).y[:, -1], phasecondition(u0, t, **params))
            return g
        return G
    
    shootingoutput = shooting2(ode, u0, phasecondition, fsolve)
    sol = solve_ivp(ode, t, y0=u0, method='RK45', args=args, max_step=max_step)


def orbit(f, u0, phasecondition, system, *params):
    """
    Solve and plot result of root finding problem.
    
    Parameters
    ----------
    f : function
        The function representing the ODE system.
    u0 : array
        Initial conditions (x0 and t values).
    phasecondition : function
        Phase condition for limit cycle.
    system : boolean
        If the ODE is a system or not.
    *params : array
        Additional parameters for ODE.
        
    Returns
    -------
    sol : array
        Solution of the BVP.
    solt : array
        Time value of the solution.
    """
    shootingsol = fsolve(shooting(f), u0, args=(phasecondition, *params), full_output=True)

    solx, solt = solve_ode(f, shootingsol[0][:-1], 0, shootingsol[0][-1], 0.01, 'RK4', system, *params)

    # plot
    def plot(ax):
        for i in range(len(solx[0])):
            ax.plot(solt, solx[:, i], label=f"State {i}")
            ax.set_title("Periodic Orbit Found Using Shooting Method for {f.__name__}")
            ax.set_xlabel("Time")
            ax.set_ylabel("dX/dt")
            ax.legend()

    if system:
        fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax2.plot(solx[:, 0], solx[:, 1])
        ax2.set_title("Phase Plane Diagram")
        ax2.set_xlabel("Prey Population")
        ax2.set_ylabel("Predator Population")
        ax2.grid(True)
        plot(ax1)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        plot(ax)

    plt.show()

    return solx, solt



def main():
    # Define the ODE system
    f = predator_prey
    # Initial conditions
    x0 = np.array([0.2, 0.2])
    # Phase condition
    pc = phase_condition
    # System
    system = True
    # Parameters
    params = (1, 0.2, 0.1)

    ppsolx, ppsolt = solve_ode(f, x0, 0, 100, 0.01, 'RK4', system, params)

    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(ppsolt, ppsolx, label='Prey Population')
    ax1.set_title("Predator-Prey Dynamics Over Time Using RK4")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Population")
    ax1.legend()

    ax2.plot(ppsolx[:, 0], ppsolx[:, 1])
    ax2.set_title("Phase Plane Diagram")
    ax2.set_xlabel("Prey Population")
    ax2.set_ylabel("Predator Population")
    ax2.grid(True)
    plt.show()

if __name__ == "__main__":
    main()




    
        

   
        
        



    
    