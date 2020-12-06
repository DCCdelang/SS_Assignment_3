
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
    print(cost)
    return cost

def random_small_step(x):
    return x + 0.1 * (random.random() - .2)

def random_big_step(x):
    return x + 10 * (random.random() - .2)


route = random_state
calculate_cost(route)



# local_opt = frigidum.sa(random_start=random_state,
#            objective_function=calculate_cost,
#            neighbours=[random_small_step, random_big_step], 
#            copy_state=frigidum.annealing.naked,
#            T_start=10**5,
#            alpha=.92,
#            T_stop=0.001,
#            repeats=10**2
#            )
