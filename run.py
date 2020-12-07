from func_dante import *


tsp_file = "TSP-Configurations/eil51.tsp.txt"
T = 500
<<<<<<< HEAD
scheme = "log" # Choose "lin" or "log"
=======
scheme = "std" # "lin", "log" or "std"
>>>>>>> e08f1a4506d7184f3999a1ad1b668cabef2b4b3f
N_sim = 1

routes, costs = run_annealing(tsp_file, T, scheme, N_sim)
route = routes[0]
<<<<<<< HEAD
print(route)
# plot_route(tsp_file,route)
# plt.show()

# routes, costs = run_two_opt(tsp_file, N_sim)
# route = routes[0]
# print(route)
# plot_route(tsp_file,route)
# plt.show()
=======
# print(route)
plot_route(tsp_file,route)
plt.show()

routes, costs = run_two_opt(tsp_file, N_sim)
route = routes[0]
# print(route)
plot_route(tsp_file,route)
plt.show()
>>>>>>> e08f1a4506d7184f3999a1ad1b668cabef2b4b3f
