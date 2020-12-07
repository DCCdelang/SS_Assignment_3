"""
Main file with all functions needed for run.py
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random import RandomState

rs = RandomState(420)
random.seed(420)

def make_matrix(tsp_file):
    """"
    Creates an adjacency matrix based on the tsp file
    """ 
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
def cost_change(adjacency_matrix, n1, n2, n3, n4):
    """
    Calculates change of cost for two_opt function
    """
    return adjacency_matrix[n1][n3] + adjacency_matrix[n2][n4] - adjacency_matrix[n1][n2] - adjacency_matrix[n3][n4]

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
                if cost_change(adjacency_matrix, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    return best

def calculate_cost(route, adjacency_matrix):
    '''
    Returns the cost of the current route based on adjacency matrix
    '''
    route_shifted = np.roll(route,1)
    cost = np.sum(adjacency_matrix[route, route_shifted])
    st_dev = np.std(adjacency_matrix[route, route_shifted])
    return st_dev, cost

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

def tsp_annealing(T, scheme, route, adjacency_matrix):
    """
    Annealing function with different parameter possibilities
    """
    best = route
    MC = 0
    chain_length = 0
    while T > .01 and chain_length < 1000:
        # Sample city from route
        index1, index2 = np.random.randint(0,len(route),size=2)
        sd0, cost0 = calculate_cost(route,adjacency_matrix)

        route[index1], route[index2] = route[index2], route[index1]
        sd1, cost1 = calculate_cost(route,adjacency_matrix)

        # Adjust temperature
        if scheme == "lin":
            T = T*0.99
        if scheme == "log":
            a = 10
            b = 200
            T = a/np.log(chain_length+b)
        if scheme == "std":
            delta = .01
            T = T / (1 + ((np.log(1+delta)* T) / (3 * sd0)))

        if cost1 < cost0:
            cost0 = cost1
            chain_length += 1
        else:
            U = rs.uniform()
            if U < np.exp((cost0-cost1)/T):
                cost0 = cost1
                MC += 1
                chain_length += 1
            else:
                route[index1], route[index2] = route[index2], route[index1]
        route = best
    return best

def run_annealing(tsp_file, T, scheme, N_sim):
    """
    Run function for annealing function
    """
    best_routes = []
    calculate_costs = []
    adjacency_matrix = make_matrix(tsp_file)

    for _ in range(N_sim):
        x = list(range(len(adjacency_matrix)))
        init_route = random.sample(x,len(x))
        best_route = tsp_annealing(T, scheme, init_route, adjacency_matrix)

        calculate_costs.append(calculate_cost(best_route,adjacency_matrix)[1])
        best_routes.append(best_route)
    return best_routes, calculate_costs

def plot_route(tsp_file,route):
    """
    Plot function to show route
    """
    node_list = []
    with open(tsp_file,"r") as reader:
        for line in reader:
            if line[0].isdigit() == True:
                node_list.append([int(x) for x in line.split()])
    for i in range(len(node_list)):
        plt.scatter(node_list[i][1],node_list[i][2],c="r")
    for i in range(len(route)-1):
        node1 = node_list[route[i]]
        node2 = node_list[route[i+1]]
        plt.plot([node1[1],node2[1]],[node1[2],node2[2]],"b")
    node1 = node_list[route[-1]]
    node2 = node_list[route[0]]
    plt.plot([node1[1],node2[1]],[node1[2],node2[2]], "b")
    plt.show()
