"""
Simulated Annealing for TSP
Primarily based on: 
https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from make_matrix import make_matrix

tsp_file = "TSP-Configurations/eil51.tsp.txt"


def calculate_cost(route, adjacency_matrix):
    '''
    Returns the cost of the current route
    '''
    route_shifted = np.roll(route,1)
    cost = np.sum(adjacency_matrix[route, route_shifted])
    return cost


def two_opt_annealing(route, adjacency_matrix):
    best = route
    T = 500 # initial T
    a = 10
    b = 200
    MC = 0
    changes = 0
    while T > 1.35:
        # Cooling scheme T
        # T = T*0.99
        T = a/np.log(changes+b)


        # Sample city from route
        index1, index2 = np.random.randint(0,len(route),size=2)
        cost0 = calculate_cost(route,adjacency_matrix)

        route[index1], route[index2] = route[index2], route[index1]
        cost1 = calculate_cost(route,adjacency_matrix)

        if cost1 < cost0:
            cost0 = cost1
            changes += 1
        else:
            U = np.random.uniform()
            if U < np.exp((cost0-cost1)/T):
                # print(T, np.exp((cost0-cost1)/T))
                cost0 = cost1
                MC += 1
                changes += 1
            else:
                route[index1], route[index2] = route[index2], route[index1]
        route = best
    print("MC =", MC, "Changes =", changes)
    return best

def run_two_opt_anneal(T,scheme, N_sim):
    best_routes = []
    calculate_costs = []
    N_sim = 1
    adjacency_matrix = make_matrix(tsp_file)

    for _ in range(N_sim):
        x = list(range(len(adjacency_matrix)))
        init_route = random.sample(x,len(x))
        print(init_route)
        print(calculate_cost(init_route,adjacency_matrix))

        adjacency_matrix = make_matrix(tsp_file)
        best_route = two_opt_annealing(init_route, adjacency_matrix)

        print(best_route)
        print(calculate_cost(best_route,adjacency_matrix))
        calculate_costs.append(calculate_cost(best_route,adjacency_matrix))
        best_routes.append(best_routes)
    return best_route

def plot_route(tsp_file,route):
    node_list = []
    with open(tsp_file,"r") as reader:
        for line in reader:
            if line[0].isdigit() == True:
                node_list.append([int(x) for x in line.split()])
    for i in range(len(node_list)):
        plt.scatter(node_list[i][1],node_list[i][2])
    for i in range(len(route)-1):
        node1 = node_list[route[i]]
        node2 = node_list[route[i+1]]
        plt.plot([node1[1],node2[1]],[node1[2],node2[2]])
    node1 = node_list[route[-1]]
    node2 = node_list[route[0]]
    plt.plot([node1[1],node2[1]],[node1[2],node2[2]])
    plt.show()

route = run_two_opt_anneal()
plot_route(tsp_file,route)