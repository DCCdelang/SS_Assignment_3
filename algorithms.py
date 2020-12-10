"""
Main file with all functions needed for run.py
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random import RandomState

from make_matrix import make_matrix

rs = RandomState(420)
random.seed(420)


def cost_change(adjacency_matrix, n1, n2, n3, n4):
    """
    Calculates change of cost for two_opt function
    """
    return adjacency_matrix[n1][n3] + adjacency_matrix[n2][n4] - \
        adjacency_matrix[n1][n2] - adjacency_matrix[n3][n4]

def calculate_cost(route, adjacency_matrix):
    """
    Returns the cost of the current route based on adjacency matrix
    """
    route_shifted = np.roll(route,1)
    cost = np.sum(adjacency_matrix[route, route_shifted])
    st_dev = np.std(adjacency_matrix[route, route_shifted])
    return st_dev, cost


# Primarly based on:
# https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
def two_opt(route, adjacency_matrix, max_chain_length):
    """
    Calculates the best route using greedy two_opt
    """
    cost_list = []
    chain = 0
    while chain < max_chain_length:
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                chain += 1

                if j - i == 1: continue

                cost_list.append(calculate_cost(route,adjacency_matrix)[1])

                if cost_change(adjacency_matrix, route[i - 1], route[i], \
                    route[j - 1], route[j]) < -0.001:
                    route[i:j] = route[j - 1:i - 1:-1]

                if chain == max_chain_length:
                    return route, cost_list

    return route, cost_list


def run_two_opt(tsp_file, N_sim, max_chain_length):
    """
    Run function for greedy two_opt function
    """
    best_routes = []
    costs = []
    adjacency_matrix = make_matrix(tsp_file)
    cost_lists = []

    for _ in range(N_sim):
        x = list(range(len(adjacency_matrix)))
        init_route = random.sample(x,len(x))
        best_route, cost_list = two_opt(init_route, adjacency_matrix, max_chain_length)
        
        best_routes.append(best_route)
        costs.append(calculate_cost(best_route,adjacency_matrix)[1])
        cost_lists.append(cost_list)

    return best_routes, costs, cost_lists


def two_opt_annealing(T, scheme, route, adjacency_matrix, max_chain_length, c):
    """
    Calculates the best route using greedy two_opt
    """
    best = route.copy()
    cost_list, T_list = [], []
    accept_list = [[],[]]

    chains = 0

    while T > 0:
        for i in range(1, len(route) - 2):
            # Adjust temperature
            if scheme == "lin":
                T = T*c
            if scheme == "log":
                C = 1
                T = C/np.log(chains)
            elif scheme == "std":
                delta = .01
                T = T / (1 + ((np.log(1+delta)* T) / (3 * sd0)))
        

            for j in range(i + 1, len(route)):
                chains += 1
                
                if j - i == 1: continue

                cost_list.append(calculate_cost(best,adjacency_matrix)[1])
                T_list.append(T)

                if cost_change(adjacency_matrix, best[i - 1], best[i], \
                    best[j - 1], best[j]) < -0.001:
                    best[i:j] = best[j - 1:i - 1:-1]
                else:
                    temp = best.copy()
                    sd0, cost0 = calculate_cost(temp,adjacency_matrix)
                    temp[i:j] = temp[j - 1:i - 1:-1]
                    _, cost1 = calculate_cost(temp,adjacency_matrix)
                    U = rs.uniform()
                    if U < np.exp((cost0-cost1)/T):
                        accept_list[1].append(np.exp((cost0-cost1)/T))
                        accept_list[0].insert(0,T)
                        best[i:j] = best[j - 1:i - 1:-1]
                
                if chains > max_chain_length:
                    return best, cost_list

        route = best.copy()
    return best, cost_list

def run_two_opt_annealing(tsp_file, T, scheme, N_sim, max_chain_length=100000, c=.95):
    """
    Run function for annealing function
    """
    # create empty lists for to be stored values
    best_routes, costs, cost_lists = [], [], []

    adjacency_matrix = make_matrix(tsp_file)

    for _ in range(N_sim):
        # generate random initial route
        x = list(range(len(adjacency_matrix)))
        init_route = random.sample(x,len(x))
        
        # find best route with SA algorithm
        best_route, cost_list = two_opt_annealing(T, scheme, init_route, adjacency_matrix, max_chain_length, c)

        # append all values from simulation to lists
        costs.append(calculate_cost(best_route,adjacency_matrix)[1])
        best_routes.append(best_route)
        cost_lists.append(cost_list)

    return best_routes, costs, cost_lists
