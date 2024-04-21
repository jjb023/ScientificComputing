from ActualODEs import *

# Example parameters for the predator-prey system
params = (0.1, 0.02, 0.3)  # a, b, d

# Initial conditions: x0 (initial prey population), y0 (initial predator population)
initial_conditions = [0.5, 0.5]

# Simulation time
start_time = 0
end_time = 200
time_step = 0.1

# Solving the predator-prey system using the RK4 method
times, populations = solve_ode(predator_prey, initial_conditions, start_time, end_time, time_step, method='rk4', params=params)
plot_ode(times, populations, 'Prey', 'Predator', title='Predator-Prey System')
# `populations` will be an array where the first column is the prey population and the second column is the predator population over time