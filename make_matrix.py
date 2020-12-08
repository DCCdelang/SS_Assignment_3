import numpy as np
import matplotlib.pyplot as plt

def make_matrix(tsp_file):
    """"
    Creates a matrix with euclidian distances 
    based on the tsp.text files
    """ 
    # Extracting node coordinates from tsp file
    node_list = []
    with open(tsp_file,"r") as reader:
        for line in reader:
            line = line.split()
            if line[0].isdigit() == True:
                node_list.append([int(x) for x in line])

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