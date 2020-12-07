from func_dante import *


tsp_file = "TSP-Configurations/eil51.tsp.txt"
T = 10000
scheme = "lin" # "lin", "log" or "std"
N_sim = 1

routes, costs = run_annealing(tsp_file, T, scheme, N_sim)
route = routes[0]
plot_route(tsp_file,route)
# plt.show()
print('lin scheme =', costs[-1])

scheme = "log" # "lin", "log" or "std"
routes, costs = run_annealing(tsp_file, T, scheme, N_sim)
route = routes[0]
plot_route(tsp_file,route)
# plt.show()
print('log scheme =', costs[-1])

scheme = "std" # "lin", "log" or "std"
routes, costs = run_annealing(tsp_file, T, scheme, N_sim)
route = routes[0]
plot_route(tsp_file,route)
print('std scheme =', costs[-1])

routes, costs = run_two_opt(tsp_file, N_sim)
route = routes[0]
print('2opt =', costs[-1])

plot_route(tsp_file,route)
plt.show()
