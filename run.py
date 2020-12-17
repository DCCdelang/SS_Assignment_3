from algorithms import *
from plot_route import plot_route

import pandas as pd
import time



tsp_file = "TSP-Configurations/a280.tsp.txt"
N_sim = 10
max_chain_length = 100000

t0 = time.time()

# T = 10000
# scheme = "expo" # "expo", "log" or "std"
# routes, costs = run_annealing_2(tsp_file, T, scheme, N_sim, max_chain_length)
# route = routes[0]
# print('scheme =', scheme, "Cost =", costs[-1])
# plot_route(tsp_file,route)

# T = 10000

# scheme = "log" # "expo", "log" or "std"
# routes, costs = run_annealing(tsp_file, T, scheme, N_sim, max_chain_length)
# route = routes[0]
# print('log scheme =', costs[-1])
# plot_route(tsp_file,route)

# T = 10000
# scheme = "log" # "expo", "log" or "std"
# routes, costs = run_annealing(tsp_file, T, scheme, N_sim, max_chain_length)
# route = routes[0]
# print('log scheme =', costs[-1])
# plot_route(tsp_file,route)

# T = 10000
# scheme = "std" # "expo", "log" or "std"
# routes, costs = run_annealing(tsp_file, T, scheme, N_sim, max_chain_length)
# route = routes[0]
# print('std scheme =', costs[-1])
# plot_route(tsp_file,route)

# routes, costs = run_two_opt(tsp_file, N_sim)
# route = routes[0]
# print('2opt =', costs[-1])
# plot_route(tsp_file,route)

# plt.show()
