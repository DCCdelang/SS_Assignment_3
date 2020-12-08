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
    print(cost)
    st_dev = np.std(adjacency_matrix[route, route_shifted])
    return st_dev, cost


# Primarly based on:
# https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
def two_opt(route, adjacency_matrix):
    """
    Calculates the best route using greedy two_opt
    """
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(adjacency_matrix, best[i - 1], best[i], \
                    best[j - 1], best[j]) < -0.001:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
    return best

def run_two_opt(tsp_file, N_sim):
    """
    Run function for greedy two_opt function
    """
    best_routes = []
    calculate_costs = []
    adjacency_matrix = make_matrix(tsp_file)

    for _ in range(N_sim):
        x = list(range(len(adjacency_matrix)))
        init_route = random.sample(x,len(x))
        best_route = two_opt(init_route, adjacency_matrix)
        
        calculate_costs.append(calculate_cost(best_route,adjacency_matrix)[1])
        best_routes.append(best_route)

    return best_routes, calculate_costs

def tsp_annealing(T, scheme, route, adjacency_matrix, max_chain_length):
    """
    Annealing function with different parameter possibilities
    """
    best = route
    iterations = 1
    chain_length = 0
    cost_list = []
    T_list = []
    while T > 0.01 and iterations < max_chain_length:
        # Sample city from route
        index1, index2 = np.random.randint(0,len(route),size=2)
        sd, cost0 = calculate_cost(route,adjacency_matrix)
        cost_list.append(cost0)

        route[index1], route[index2] = route[index2], route[index1]
        _, cost1 = calculate_cost(route,adjacency_matrix)

        iterations += 1

        # Adjust temperature
        if scheme == "lin":
            T = T*0.975
        if scheme == "log":
            C = 1
            T = C / np.log(chain_length)
        if scheme == "std":
            delta = .1
            T = T / (1 + ((np.log(1+delta)* T) / (3 * sd)))

        T_list.insert(0,T)

        # Metropolis step   
        if cost1 < cost0:
            cost0 = cost1
            chain_length += 1
        else:
            U = rs.uniform()
            if U < np.exp((cost0-cost1)/T):
                # print(T,np.exp((cost0-cost1)/T))
                cost0 = cost1
                chain_length += 1
            else:
                route[index1], route[index2] = route[index2], route[index1]
        route = best
    print("Max iterations:", iterations)
    # plt.plot(T_list,cost_list)
    # plt.plot(range(len(cost_list)),cost_list)
    # plt.show()
    return best

def run_annealing(tsp_file, T, scheme, N_sim, max_chain_length):
    """
    Run function for annealing function
    """
    best_routes = []
    calculate_costs = []
    adjacency_matrix = make_matrix(tsp_file)

    for _ in range(N_sim):
        x = list(range(len(adjacency_matrix)))
        init_route = random.sample(x,len(x))
        best_route = tsp_annealing(T, scheme, init_route, adjacency_matrix, max_chain_length)

        calculate_costs.append(calculate_cost(best_route,adjacency_matrix)[1])
        best_routes.append(best_route)
    return best_routes, calculate_costs


def two_opt_annealing(T, scheme, route, adjacency_matrix, max_chain_length):
    """
    Calculates the best route using greedy two_opt
    """
    best = route
    iterations = 1
    cost_list = []
    T_list = []
    
    sd, cost0 = calculate_cost(route, adjacency_matrix)

    while T > 0.01 and iterations < max_chain_length:
        for i in range(1, len(route) - 2):
            # Adjust temperature
            T_list.append(T)
            if scheme == "lin":
                T = T * 0.975
            if scheme == "log":
                a = 1
                b = 100
                T = a/np.log(iterations + b)
            if scheme == "std":
                delta = .1
                T = T / (1 + ((np.log(1+delta)* T) / (3 * sd)))
                

            for j in range(i + 1, len(route)):
                iterations += 1
                if j - i == 1: continue
                if cost_change(adjacency_matrix, best[i - 1], best[i], \
                    best[j - 1], best[j]) < -0.001:
                    best[i:j] = best[j - 1:i - 1:-1]
                else:
                    temp = best
                    sd, cost0 = calculate_cost(temp,adjacency_matrix)
                    temp[i:j] = temp[j - 1:i - 1:-1]
                    _, cost1 = calculate_cost(temp,adjacency_matrix)
                    U = rs.uniform()
                    if U < np.exp((cost0-cost1)/T):
                        print('T:', T, 'chance', np.exp((cost0-cost1)/T))
                        best[i:j] = best[j - 1:i - 1:-1]

                cost_list.append(cost0)


        route = best

    plt.plot(range(len(cost_list)),cost_list)
    plt.show()

    return best

def run_two_opt_annealing(tsp_file, T, scheme, N_sim, max_chain_length):
    """
    Run function for annealing function
    """
    best_routes = []
    calculate_costs = []
    adjacency_matrix = make_matrix(tsp_file)

    for _ in range(N_sim):
        x = list(range(len(adjacency_matrix)))
        init_route = random.sample(x,len(x))
        best_route = two_opt_annealing(T, scheme, init_route, adjacency_matrix, max_chain_length)

        calculate_costs.append(calculate_cost(best_route,adjacency_matrix)[1])
        best_routes.append(best_route)
    return best_routes, calculate_costs
