import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import adjustment
import mSBD
import graph

def constructive_FD(G, X, Y):
	'''
	Find the set Z_ii in time O(n + m).
	
	G : NetworkX DiGraph (Directed Acyclic Graph)
	X : set of nodes
	Y : set of nodes
	Z_i : set of nodes
	'''
	V = set(G.nodes())
	U_set = set([n for n in G.nodes() if n.startswith('U')])
	Z_i = V - U_set  - set(Y) - set(X) - set(graph.find_parents(graph.G_cut_outgoing_edges(G,X),X)) - set( graph.find_c_components(graph.G_cut_outgoing_edges(G,X),X) )

	visited = {v: {'inc': False, 'out': False} for v in V}
	continue_later = {v: False for v in V}
	forbidden = {v: (v not in Z_i) for v in V} 
	
	def visit(G, V, edgetype):
		visited[V][edgetype] = True
		forbidden[V] = True
		if V not in X:
			if edgetype == 'inc':
				for W in G.successors(V):  # Children of V
					if not visited[W]['inc']:
						visit(G, W, 'inc')
			if edgetype == 'out':
				for W in G.predecessors(V):  # Parents of V
					if not visited[W]['out']:
						if forbidden[W]:
							visit(G, W, 'out')
						else:
							continue_later[W] = True
		if continue_later[V] and not visited[V]['out']:
			visit(G, V, 'out')
	
	for Y_node in Y:
		if not visited[Y_node]['out']:
			visit(G, Y_node, 'out')
	
	Z_ii = list( V - {v for v in V if forbidden[v]} )
	if len(Z_ii) == 0:
		return False
	C = set( adjustment.construct_minimum_adjustment_set(G,X,Z_ii) ) - set(X) - set(Y)
	CX = list(set(X).union(set(C)))
	ZC = list(C.union(set(Z_ii)))
	X_C = list( set(X) - set(graph.find_ancestor(graph.G_cut_incoming_edges(G,Z_ii), C)) )

	condition1 = graph.is_d_separated(graph.G_cut_outgoing_edges(G, X), X, Z_ii, C)
	condition2 = graph.is_d_separated(graph.G_cut_incoming_edges(graph.G_cut_outgoing_edges(G, Z_ii), X), Y, Z_ii, CX)
	condition3 = graph.is_d_separated(graph.G_cut_incoming_edges(graph.G_cut_incoming_edges(G, Z_ii), X_C), Y, X, ZC)
	condition4 = adjustment.check_adjustment_criterion(G, Z_ii, Y, CX)

	if condition1 and condition2 and condition3 and condition4: 
		FD_true_false = True 
		CZ_dict = {"Z": Z_ii, "C": list(C)}
		return CZ_dict
	return False 


def constructive_minimum_FD(G, X, Y):
	'''
	Find the minimal front-door adjustment set Z_min with I ⊆ Z_min ⊆ R or ⊥ if no FD set exists.
	
	G : NetworkX DiGraph (Directed Acyclic Graph)
	X : set of nodes
	Y : set of nodes
	I : set of nodes
	R : set of nodes
	'''
	# Step 1: Compute Z(ii)
	ZC = constructive_FD(G, X, Y)
	if ZC == False:
		return False  # Return ⊥ if no FD set exists
	else: 
		Z_ii = set( ZC['Z'] ) 

	R = set([node for node in nx.topological_sort(G) if not node.startswith('U')])
	I = set(ZC['C'])
	
	def get_parents_and_paths_to_Y(G, Z_ii, Y):
		# Get parents and nodes with paths to Y
		ZAn_candidates = set(graph.find_ancestor(G,Y)).intersection(set(Z_ii))
		Z_An = set()
		parents_Y = set(graph.find_parents(G, Y))
		for v in ZAn_candidates:
			if v in parents_Y:
				Z_An.add(v)
				continue
			exclude_nodes = set(X).union(Z_ii - {v})
			for y in Y:
				inbetween = set(graph.find_ancestor(G,[y])).intersection(set(graph.find_descendant(G,[v])))
				if len(inbetween.intersection(exclude_nodes)) == 0:
					Z_An.add(v)
					continue
		return Z_An
	
	def get_nodes_with_paths_to_X(G, Z_An, X):
		# return set(graph.find_descendant(G,X)).intersection(set(Z_An))
		# Get nodes with paths to X
		Z_XY_candidates = set(graph.find_descendant(G,X)).intersection(set(Z_An))
		Z_XY = set()
		for v in Z_An:
			exclude_nodes = set(Z_An) - {v}
			for x in X:
				if nx.has_path(G, x, v):
					inbetween = set(graph.find_ancestor(G,[v])).intersection(set(graph.find_descendant(G,[x])))
					if len(inbetween.intersection(exclude_nodes)) == 0:
						Z_XY.add(v)
						continue
		return Z_XY
	
	# Step 2: Compute Z_An
	Z_An = get_parents_and_paths_to_Y(G, Z_ii, Y)
	if len(Z_An) == 0:
		Z_An = Z_ii
	
	# Step 3: Compute Z_XY
	Z_min = get_nodes_with_paths_to_X(G, Z_An, X)
	if len(Z_min) == 0:
		Z_min = Z_An

	C_min = list(set( adjustment.construct_minimum_adjustment_set(G,Z_min,Y) ) - set(X) - set(Y))
	
	ZC['Z'] = list(Z_min )
	ZC['C'] = C_min
	
	return ZC


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