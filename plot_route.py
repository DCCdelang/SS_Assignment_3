import matplotlib.pyplot as plt 

def plot_route(tsp_file,route):
    """
    Plot function to show route
    """
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