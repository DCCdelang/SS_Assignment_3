
import frigidum 
import numpy as np
import random

# from frigidum.examples import tsp

from get_distances import get_distances

distances = get_distances('TSP-Configurations/eil51.tsp.txt')
total_nodes = len(distances[0])

def random_state():
	"""
	Returns a random state
	"""
	state = np.arange(0,total_nodes)
	np.random.shuffle(state)
	return state

def calculate_cost(route):
    '''
    Returns the cost of the current route
    '''
    route_shifted = np.roll(route,1)
    cost = np.sum(distances[route, route_shifted])
    return cost

def random_change(route):
    index = np.random.randint(0, total_nodes) % total_nodes 
    swap = np.random.randint(0, total_nodes) % total_nodes 
    temp = route[index]
    route[index] = route[index - swap % total_nodes]
    route[index - swap % total_nodes] = temp 
    return route

def random_big_step(x):
    return x + np.random.randint(-10, 10) % total_nodes 

# # tests
# route = random_state()
# calculate_cost(route)



local_opt = frigidum.sa(random_start=random_state,
           objective_function=calculate_cost,
           neighbours=[random_change], 
           copy_state=frigidum.annealing.naked,
           T_start=10**5,
           alpha=.92,
           T_stop=0.001,
           repeats=10**2
           )
print(local_opt)