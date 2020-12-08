from algorithms import *
from plot_route import plot_route
import time

tsp_file = "TSP-Configurations/a280.tsp.txt"
N_sim = 10
max_chain_length = 5000

t0 = time.time()
T = 5000
scheme = "lin" # "lin", "log" or "std"
routes, costs, cost_lists = run_two_opt_annealing(tsp_file, T, scheme, N_sim, max_chain_length)

t1 = time.time()
print("Total time =", t1-t0)
route = routes[0]
print('scheme =', scheme, "Cost =", costs[-1])
# plot_route(tsp_file,route)

# plot costs
plt.title("Costs")
for i in range(N_sim):
    plt.plot(range(len(cost_lists[i])),cost_lists[i], label = f'attempt {i}')
plt.legend()
plt.show()



















# max_chain_length = 20000

# T = 10000
# scheme = "lin" # "lin", "log" or "std"
# routes, costs = run_annealing_2(tsp_file, T, scheme, N_sim, max_chain_length)
# route = routes[0]
# print('scheme =', scheme, "Cost =", costs[-1])
# plot_route(tsp_file,route)

# T = 10000
# scheme = "log" # "lin", "log" or "std"
# routes, costs = run_annealing(tsp_file, T, scheme, N_sim, max_chain_length)
# route = routes[0]
# print('log scheme =', costs[-1])
# plot_route(tsp_file,route)

# T = 10000
# scheme = "log" # "lin", "log" or "std"
# routes, costs = run_annealing(tsp_file, T, scheme, N_sim, max_chain_length)
# route = routes[0]
# print('log scheme =', costs[-1])
# plot_route(tsp_file,route)

# T = 10000
# scheme = "std" # "lin", "log" or "std"
# routes, costs = run_annealing(tsp_file, T, scheme, N_sim, max_chain_length)
# route = routes[0]
# print('std scheme =', costs[-1])
# plot_route(tsp_file,route)

# routes, costs = run_two_opt(tsp_file, N_sim)
# route = routes[0]
# print('2opt =', costs[-1])
# plot_route(tsp_file,route)

# plt.show()
