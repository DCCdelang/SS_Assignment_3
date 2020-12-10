'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Calculates costs for different constants for the linear cooling schedule 
and writes the results to a CSV file + plots the results.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from algorithms import *
from plot_route import plot_route

import pandas as pd
import time

def calculate_and_plot(cost_lists, N_sim):
    '''
    returns mean values for elements of lists in list
    for all different constant values of the cooling scheme
    '''
    means = []

    # calculate mean costs for all iterations for all simulations
    for j in range(len(cost_lists[0])):
        temp_list = []
        for i in range(N_sim):
            temp_list.append(cost_lists[i][j])
        means.append(np.mean(temp_list))

        # if j == len(cost_lists[0]):



    # put data in pandas df and write to csv
    df_means = pd.DataFrame(means,dtype=float)
    df_means.to_csv(f"data/cooling_rate_{c}.csv")

    # plot data
    print('c =', c, ':', means[-1])
    plt.plot(range(len(means)), means, label = f'c = .{c}')
    


# define variables
tsp_file = "TSP-Configurations/a280.tsp.txt"
N_sim = 1
max_chain_length = 100000
T = 5000
t0 = time.time()

# linear scheme
scheme = "lin" 

constants = [.2, .4, .6, .8, .975]

for c in constants:
    _, _, cost_lists = run_two_opt_annealing(tsp_file, T, scheme, N_sim, max_chain_length, c)
    calculate_and_plot(cost_lists, N_sim)

# # routes, costs = run_two_opt(tsp_file, N_sim)
# route = routes[0]
# print('2opt =', costs[-1])
# plot_route(tsp_file,route)

t1 = time.time()
print("The simulation took ", round(t1-t0), 'seconds')

plt.xlabel('MCMC (iterations)')
plt.ylabel('Cost')
plt.legend()
plt.show()

