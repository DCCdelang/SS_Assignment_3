from func_dante import *


tsp_file = "TSP-Configurations/eil51.tsp.txt"
T = 10000
scheme = "lin" # "lin", "log" or "std"
N_sim = 1

routes, costs = run_annealing(tsp_file, T, scheme, N_sim)
route = routes[0]
# print(route)
plot_route(tsp_file,route)
plt.show()

routes, costs = run_two_opt(tsp_file, N_sim)
route = routes[0]
# print(route)
plot_route(tsp_file,route)
plt.show()
