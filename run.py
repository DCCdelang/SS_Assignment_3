from func_dante import *

T = 500
scheme = "log" # "lin" or "log"
N_sim = 1
routes, costs = run_two_opt_anneal(tsp_file, T, scheme, N_sim)
route = routes[0]
print(route)
plot_route(tsp_file,route)
