"""
Main file to read out data from the tsp files
"""

import numpy as np
import matplotlib.pyplot as plt
import random

tsp_file = "TSP-Configurations/eil51.tsp.txt"


def make_matrix(tsp_file):
    # Extracting node coordinates from tsp file
    node_list = []
    with open(tsp_file,"r") as reader:
        for line in reader:
            if line[0].isdigit() == True:
                node_list.append([int(x) for x in line.split()])

    # Creating adjacency matrix
    num_node = len(node_list)
    adjacency_matrix = np.zeros((num_node,num_node))
    for node1 in range(num_node):
        for node2 in range(num_node):
            if node1 != node2:
                x = abs(node_list[node1][1] - node_list[node2][1])
                y = abs(node_list[node1][2] - node_list[node2][2])
                dist = np.sqrt(x**2+y**2)
                adjacency_matrix[node1][node2] = dist

    return adjacency_matrix

# primarly based on https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]

def two_opt(route, cost_mat):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    return best

def len_route(route,adjacency_matrix):
    length = 0
    for i in range(len(route)-1):
        length += adjacency_matrix[route[i]][route[i+1]]
    length += adjacency_matrix[route[-1]][route[0]]
    return length

def run_two_opt():
    best_routes = []
    len_routes = []
    N_sim = 300
    adjacency_matrix = make_matrix(tsp_file)

    for sol in range(N_sim):
        x = list(range(len(adjacency_matrix)))
        init_route = random.sample(x,len(x))
        # print(init_route)
        # print(len_route(init_route,adjacency_matrix))

        adjacency_matrix = make_matrix(tsp_file)
        best_route = two_opt(init_route, adjacency_matrix)
        # print(best_route)
        # print(len_route(best_route,adjacency_matrix))
        len_routes.append(len_route(best_route,adjacency_matrix))
        best_routes.append(best_routes)

    plt.plot(range(N_sim),best_routes)
    plt.show()

def two_opt_annealing(route, adjacency_matrix):
    best = route
    T = 500 # initial T
    a = 10
    b = 200
    MC = 0
    changes = 0
    while T > 1.35:
        # T = T*0.99
        # print(T)
        T = a/np.log(changes+b)

        # Sample city from route
        index1, index2 = np.random.randint(0,len(route),size=2)
        cost0 = len_route(route,adjacency_matrix)

        route[index1], route[index2] = route[index2], route[index1]
        cost1 = len_route(route,adjacency_matrix)

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

def run_two_opt_anneal():
    best_routes = []
    len_routes = []
    N_sim = 1
    adjacency_matrix = make_matrix(tsp_file)

    for _ in range(N_sim):
        x = list(range(len(adjacency_matrix)))
        init_route = random.sample(x,len(x))
        print(init_route)
        print(len_route(init_route,adjacency_matrix))

        adjacency_matrix = make_matrix(tsp_file)
        best_route = two_opt_annealing(init_route, adjacency_matrix)

        print(best_route)
        print(len_route(best_route,adjacency_matrix))
        len_routes.append(len_route(best_route,adjacency_matrix))
        best_routes.append(best_routes)

    # plt.plot(range(N_sim),best_routes)
    # plt.show()

run_two_opt_anneal()
