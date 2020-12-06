"""
Main file to read out data from the tsp files
"""

import numpy as np
import matplotlib.pyplot as plt

tsp_file = "TSP-Configurations/eil51.tsp.txt"

node_list = []

with open(tsp_file,"r") as reader:
    for line in reader:
        if line[0].isdigit() == True:
            node_list.append([int(x) for x in line.split()])

# for coordinate in node_list:
#     plt.scatter(coordinate[1],coordinate[2])
# plt.grid()
# plt.show()

# Creating adjacency matrix
init_route = [1,2,6,9,4,10]
num_node = len(node_list)
adjacency_matrix = np.zeros((num_node,num_node))
for node1 in range(num_node):
    for node2 in range(num_node):
        if node1 != node2:
            x = abs(node_list[node1][1] - node_list[node2][1])
            y = abs(node_list[node1][2] - node_list[node2][2])
            dist = np.sqrt(x**2+y**2)
            adjacency_matrix[node1][node2] = dist

print(adjacency_matrix)