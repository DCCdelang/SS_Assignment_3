
import frigidum 
import numpy as np
import random

from frigidum.examples import tsp

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

def random_swap(route):
    '''
    Get some neighbour
    '''
    r1, r2 = np.random.randint(0, total_nodes, size=2)

    # swap values
    temp = route[r1]
    route[r1] = route[r2]
    route[r2] = temp

    return route



def random_disconnect_vertices_and_fix( route ):
	"""
		Randomly Disconnects K nodes (set to -1),

		After damange of bomb, it will re-connect all affected
		nodes with a fixing method.

	"""

	bombed_route = route.copy()

	size_of_bomb = 1 + np.random.poisson( 12 )

	random_indices = np.random.choice( route , size_of_bomb, replace = False)

	bombed_route[ random_indices ] = -1

	return stochastic_greedy_reroute_missing(bombed_route)

def stochastic_greedy_reroute_missing( broken_route ):
	"""
		Greedy fix (small distances have higher chance),
		a broken route.

		It wil first create a small cycle sub-route.
		This is a incomplete_route.

		For all remaining missing nodes, it will add them
		to the route, by inserting them (reroute).

		The selection process is based on chance,
		but smaller reroute have higher chance to be picked.
	"""

	complete_nodes = np.arange(broken_route.size)
	missing_nodes = np.delete(complete_nodes, broken_route[ broken_route > -1] )

	if missing_nodes.size == broken_route.size:
		broken_route[0] = 0
		missing_nodes = missing_nodes[1:]

	"""
		Split: 
		https://stackoverflow.com/questions/38277182/splitting-numpy-array-based-on-value

	"""

	missing_idx = np.where(broken_route != -1)[0]
	list_of_parts = np.split(broken_route[missing_idx],np.where(np.diff(missing_idx)!=1)[0]+1)

	incomplete_route = tsp.stochastic_glue_enpoints(list_of_parts)[0]

	"""
		for vertex in missing_nodes, reroute ass in incomplete_route
	"""
	complete_route = tsp.stochastic_reroute_missing(incomplete_route, missing_nodes)


	return complete_route


local_opt = frigidum.sa(random_start=random_state,
           objective_function=calculate_cost,
           neighbours=[tsp.euclidian_bomb_and_fix, tsp.euclidian_nuke_and_fix, tsp.route_bomb_and_fix, tsp.random_disconnect_vertices_and_fix], 
           copy_state=frigidum.annealing.naked,
           T_start=10**5,
           alpha=.92,
           T_stop=0.001,
           repeats=10**2,
           )

print(local_opt)