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
    means,stds = [],[]

    # calculate mean costs for all iterations for all simulations
    for j in range(len(cost_lists[0])):
        temp_list = []
        for i in range(N_sim):
            temp_list.append(cost_lists[i][j])
        means.append(np.mean(temp_list))
        stds.append(np.std(temp_list))

    print("T =", T,  "Mean cost =", means[-1])

    # # put data in pandas df and write to csv
    df_means = pd.DataFrame({"Means":means,"Std":stds},dtype=float)
    df_means.to_csv(f"data/two_opt_anneal/log_temp_{T}.csv")

    # plot data
    plt.plot(range(len(means)), means, label = f'T = {T}')


# define variables
tsp_file = "TSP-Configurations/a280.tsp.txt"
N_sim = 10
max_chain_length = 100000
t0 = time.time()

# uncomment to choose scheme
# scheme = "log" 
scheme = "log" 

T_list = [100, 1000, 10000]

for T in T_list:
    _, _, cost_lists = run_two_opt_annealing(tsp_file, T, scheme, N_sim, \
    max_chain_length)
    calculate_and_plot(cost_lists, N_sim)

t1 = time.time()
print("The simulation took ", round(t1-t0), 'seconds')

plt.xlabel('MCMC (iterations)')
plt.ylabel('Cost')
plt.legend()
plt.show()
