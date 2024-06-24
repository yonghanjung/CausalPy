import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

import adjustment
import mSBD
import graph


def intercept(G, X, Y, Z):
    """
    Check if Z d-separates X from Y in the graph G.
    """
    def is_blocked(path, Z):
        for i, node in enumerate(path):
            if i == 0 or i == len(path) - 1:
                continue
            
            # Check if the node is a collider
            is_collider = G.has_edge(path[i-1], node) and G.has_edge(path[i+1], node)
            
            if is_collider:
                # Collider: blocked if not in Z and has no descendants in Z
                if node not in Z and not any(desc in Z for desc in nx.descendants(G, node)):
                    return True
            else:
                # Non-collider: blocked if in Z
                if node in Z:
                    return True
        return False

    for x in X:
        for y in Y:
            paths = nx.all_simple_paths(G, x, y)
            if all(is_blocked(path, Z) for path in paths):
                continue
            return False
    return True

# # Example usage
# G = nx.DiGraph()
# # Add edges to G to define the causal structure
# # G.add_edge('A', 'B'), etc.

# X = {'A', 'B'}
# Y = {'Y'}
# Z = {'Z'}

# print(is_d_separated(G, X, Y, Z))



def find_frontdoor(G,X,Y):
	if graph.find_parents(G,Y).intersection(set(X)):
		return False

	# candidate_Z = adjustment.proper_causal_path(G,X,Y) - set(Y) - graph.find_c_components(G,X)
	candidate_Z = set( graph.find_variables_no_inducing_path(graph.G_cut_outgoing_edges(G,X),X) ) - set(Y) - set(X) - graph.find_c_components(G,X)
	ordered_V = graph.find_topological_order(G)
	ordered_candidate_Z = [v for v in ordered_V if v in candidate_Z] 

	# Generate all possible subsets of my_list
	all_possible_Z = []
	for r in range(len(ordered_candidate_Z) + 1):
	    Z_candidate = combinations(ordered_candidate_Z, r)
	    all_possible_Z.extend(Z_candidate)

	# Convert each subset to a list (optional)
	all_possible_Z = [list(subset) for subset in all_possible_Z]
	FD_true_false = False 
	for Z in all_possible_Z:
		if Z == []:
			continue
		# if not adjustment.check_adjustment(G,X,Z):
		# 	continue
		if len(set(Z).intersection(adjustment.proper_causal_path(G,X,Y))) == 0:
			continue
		if not intercept(G,X,Y,Z):
			continue
		C = adjustment.construct_minimum_adjustment_set(G,X,Z) - set(X) - set(Y)
		# C = adjustment.construct_adjustment_set(G,X,Z) - set(X) - set(Y)
		CX = list(set(X).union(C))
		condition1 = graph.is_d_separated(graph.G_cut_incoming_edges(G, X), C, X, [])
		condition2 = adjustment.adjustment_criterion(G, Z, Y, CX)
		if condition1 and condition2: 
			FD_true_false = True 
			CZ_dict = {"Z": Z, "C": list(C)}
			return CZ_dict
		else:
			continue 
	return False 

def frontdoor_estimand(X,Y,Z,C,latex):
	CZ = list(set(C).union(set(Z)))
	XZC = list(set(CZ).union(set(X)))
	C = list(set(C))
	XC = list(set(X).union(set(C)))

	Y_val = ', '.join(Y)
	X_val = ', '.join(X)
	Z_val = ', '.join(Z)
	XC_val = ', '.join(XC)
	XZC_val = ', '.join(XZC)
	X_lower_val = ', '.join(char.lower() for char in X)

	CZ_lower_values = ', '.join(char.lower() for char in CZ)

	if len(C) == 0:
		if not latex:
			FD_adjustment = f"\u03A3_{{{CZ_lower_values}}} P({Z_val} | {XC_val}) \u03A3_{{{X_lower_val}}} P({Y_val} | {XZC_val})P({X_val})"
		else:
			FD_adjustment = f"\\sum_{{{CZ_lower_values}}} P({Z_val} \\mid {XC_val}) \\sum_{{{X_lower_val}}} P({Y_val} \\mid {XZC_val})P({X_val})"
	else:
		C_val = ', '.join(C)
		if not latex:
			FD_adjustment = f"\u03A3_{{{CZ_lower_values}}} P({Z_val} | {XC_val})P({C_val}) \u03A3_{{{X_lower_val}}} P({Y_val} | {XZC_val})P({X_val} | {C_val})"
		else:
			FD_adjustment = f"\\sum_{{{CZ_lower_values}}} P({Z_val} \\mid {XC_val})P({C_val}) \\sum_{{{X_lower_val}}} P({Y_val} \\mid {XZC_val})P({X_val} \\mid {C_val})"
	return FD_adjustment