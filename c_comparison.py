'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Calculates costs for different constants for the linear cooling schedule 
and writes the results to a CSV file + plots the results.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from algorithms import *
from plot_route import plot_route

import pandas as pd
import time

def calculate_and_plot(cost_lists, N_sim, c = '2-opt'):
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
            print(i)
        means.append(np.mean(temp_list))

    
    '''uncomment to save csv'''
    # put data in pandas df and write to csv
    if c == '2-opt':
        df_means = pd.DataFrame(means,dtype=float)
        df_means.to_csv(f"data/{c}.csv")
    else:
        df_means = pd.DataFrame(means,dtype=float)
        df_means.to_csv(f"data/{c}.csv")

    # plot data
    print('type =', c, ':', means[-1])
    plt.plot(range(len(means)), means, label = f'c = .{c}')
    


# define variables
tsp_file = "TSP-Configurations/a280.tsp.txt"
N_sim = 10
max_chain_length = 1e5
T = 5000
t0 = time.time()

# linear scheme
scheme = "lin" 

constants = [.9]

# plot 2opt costs
_, _, cost_lists_2opt = run_two_opt(tsp_file, N_sim, max_chain_length)
calculate_and_plot(cost_lists_2opt, N_sim)

# plot sim annealing costs
for c in constants:
    _, costs, cost_lists = run_two_opt_annealing(tsp_file, T, scheme, N_sim, max_chain_length, c)
    calculate_and_plot(cost_lists, N_sim, c)

t1 = time.time()
print("The simulation took ", round(t1-t0), 'seconds')

plt.xlabel('MCMC (iterations)', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.legend(fontsize=14)
plt.tick_params(labelsize=14)
plt.show()

