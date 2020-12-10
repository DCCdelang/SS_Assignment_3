from algorithms import calculate_cost
from make_matrix import make_matrix
import numpy as np
import matplotlib.pyplot as plt

def check_opt():
    tsp_file = "TSP-Configurations/a280.opt.tour.txt"
    node_list = []
    with open(tsp_file,"r") as reader:
        for line in reader:
            line = line.split()
            if line[0].isdigit() == True:
                node_list.append(int(line[0])-1)
    print(len(node_list))
    route = node_list.copy()
    tsp_file = "TSP-Configurations/a280.tsp.txt"
    adjacency_matrix = make_matrix(tsp_file)
    print(np.shape(adjacency_matrix))
    _,cost = calculate_cost(node_list,adjacency_matrix)
    print("Shortest a280 route =",cost)
    
    # Shortest a280 route = 2586.76964756316

    node_list = []
    with open(tsp_file,"r") as reader:
        for line in reader:
            line = line.split()
            if line[0].isdigit() == True:
                node_list.append([int(x) for x in line])
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