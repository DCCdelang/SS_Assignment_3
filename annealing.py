import numpy as np
import random

from get_distances import get_distances

dist_matrix = get_distances('TSP-Configurations/eil51.tsp.txt')
total_nodes = len(dist_matrix[0])


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
    cost = np.sum(dist_matrix[route, route_shifted])
    return cost

initial_route = random_state()
initial_cost = calculate_cost(initial_route)

T = 30
factor = .99 

initial_T = T


