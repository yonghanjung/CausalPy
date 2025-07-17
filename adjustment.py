import networkx as nx
import graph 

def proper_causal_path(G, X, Y):
	'''
	Find the proper causal path set: (DE_{G_{bar(X)}}(X) setminus X ) intersect AN_{G_{underline(X)}}(Y)

	Parameters:
	G (nx.DiGraph): The original directed graph.
	X (list or set): Nodes to consider for descendants and edge removals.
	Y (list or set): Nodes to consider for ancestors.

	Returns:
	set: Nodes that are descendants of X (excluding X) and ancestors of Y.
	'''

	# Step 1: Remove incoming edges to X
	Gi = graph.G_cut_incoming_edges(G, X)

	# Step 2: Remove outgoing edges from X
	Go = graph.G_cut_outgoing_edges(G, X)

	# Step 3: Find descendants of X in Gi
	de_X = graph.find_descendant(Gi, X)  # Assuming descendant() is defined

	# Step 4: Find ancestors of Y in Go
	an_Y = graph.find_ancestor(Go, Y)  # Assuming ancestor() is defined

	# Step 5: Return the intersection, excluding X
	return list((set(de_X) - set(X)) & set(an_Y))


def descedent_proper_causal_path(G, X, Y):
	pcp = proper_causal_path(G,X,Y)
	return graph.find_descendant(G,pcp)


def check_backdoor_criterion(G, X, Y, Z):
	'''
	Check if Z satisfies the Back-door Criterion relative to X and Y in graph G.

	Parameters:
	G (nx.DiGraph): The original directed graph.
	X (list): Treatment variables.
	Y (list): Outcome variables.
	Z (list): List of variables to be checked for the Back-door Criterion.

	Returns:
	bool: True if Z satisfies the Back-door Criterion, False otherwise.
	'''

	# Step 1: Check if any element in Z is a descendant of any node in X
	descendants_of_X = graph.find_descendant(G, X)
	if any(z in descendants_of_X for z in Z):
		return False  # Z contains a descendant of X

	# Step 2: Check if Z d-separates X and Y in G with outgoing edges from X removed
	G_modified = graph.G_cut_outgoing_edges(G, X)
	if graph.is_d_separated(G_modified, X, Y, Z):
		return True  # Z d-separates X and Y in the modified graph

	return False

def proper_backdoor_graph(G, X, Y):
	'''
	Modify the graph G by removing edges from nodes in X to nodes identified by proper causal path (pcp).

	Parameters:
	G (nx.DiGraph): The original directed graph.
	X (list): Set of treatment variables.
	Y (list): Set of outcome variables.

	Returns:
	nx.DiGraph: The modified graph with specific edges removed.
	'''
	# Make sure that X and Y are lists 
	X = list(X)
	Y = list(Y)

	# Assuming proper_causal_path(G, X, Y) is defined and returns a set of nodes
	pcp = proper_causal_path(G, X, Y)
	
	# Create a copy of G to avoid modifying the original graph
	G_modified = G.copy()

	# Iterate over each node in X and remove outgoing edges to nodes in pcp
	for x_node in X:
		for y_node in pcp:
			if G_modified.has_edge(x_node, y_node):
				G_modified.remove_edge(x_node, y_node)

	return G_modified

def check_adjustment_criterion(G, X, Y, Z):
	'''
	Check if Z satisfies the adjustment criterion relative to (X,Y) in G

	Parameters:
	G (nx.DiGraph): The directed graph representing the causal structure.
	X (list): Set of treatment variables.
	Y (list): Set of outcome variables.
	Z (list): Set of covariates

	Returns:
	bool: True if if Z satisfies the adjustment criterion relative to (X,Y) in G
	'''
	
	G_pbd = proper_backdoor_graph(G, X, Y)
	dpcp = descedent_proper_causal_path(G,X,Y)
	if any(z in dpcp for z in Z):
		return False  # Z contains a descendant of X

	if graph.is_d_separated(G_pbd, X, Y, Z):     	
		return True  # Z d-separates X and Y in the modified graph

	return False 

def construct_adjustment_set(G, X, Y):
	'''
	Construct an adjustment set for estimating the causal effect of X on Y.

	Parameters:
	G (nx.DiGraph): The directed graph representing the causal structure.
	X (list): Set of treatment variables.
	Y (list): Set of outcome variables.

	Returns:
	set: The set of nodes suitable for adjustment.
	'''

	# Assuming descedent_proper_causal_path(G, X, Y) is defined
	X_set = set(X)
	Y_set = set(Y)
	dpcp = descedent_proper_causal_path(G, X, Y)
	dpcp_set = set(dpcp)

	# Assuming ancestor(G, nodes) is defined
	ancestors_XY = graph.find_ancestor(G, X_set.union(Y_set))
	ancestors_XY_set = set(ancestors_XY)

	# Construct the adjustment set
	adjustment_set = ancestors_XY_set - (X_set.union(Y_set).union(dpcp_set))
	return list(adjustment_set)

def check_admissibility(G, X, Y):
	'''
	Check if P(Y | do(X)) can be represented as a back-door adjustment.

	Parameters:
	G (nx.DiGraph): The original directed graph.
	X (list): Treatment variables.
	Y (list): Outcome variables.

	Returns:
	bool: True if P(Y | do(X)) can be represented as a back-door adjustment.
	'''
	adjustment_Z = construct_adjustment_set(G, X, Y)
	if check_adjustment_criterion(G, X, Y, adjustment_Z):
		return True
	return False 

def adjustment_estimand(X,Y,Z,latex):
	'''
	Generate the back-door adjustment formula "sum_{z}P(y | x,z)P(z)". 

	Parameters:
	G (nx.DiGraph): The original directed graph.
	X (list): Treatment variables.
	Y (list): Outcome variables.
	Z (list): Covariate variables 
	latex (bool): True if the output is in the latex syntax.

	Returns:
	string: "sum_{z}P(y | x,z)P(z)"
	'''
	Z = list(set(Z))
	Z_val = ', '.join(Z)
	Z_lower_val = ', '.join(char.lower() for char in Z)

	Y_val = ', '.join(Y)
	X_val = ', '.join(X)
	XZ = list(set(X).union(set(Z)))
	XZ_val = ', '.join(XZ)

	if not latex:
		if len(Z) == 0:
			adjustment_estimand = f"P({Y_val} | {X_val})"
		else:
			adjustment_estimand = f"\u03A3_{{{Z_lower_val}}}P({Y_val} | {XZ_val}) P({Z_val})"
	else:
		if len(Z) == 0:
			adjustment_estimand = f"P({Y_val} \\mid {X_val})"
		else:
			adjustment_estimand = f"\\sum_{{{Z_lower_val}}}P({Y_val} \\mid {XZ_val}) P({Z_val})"
	return adjustment_estimand

def construct_minimum_adjustment_set(G,X,Y):
	'''
	Construct an minimum adjustment set for estimating the causal effect of X on Y.

	Parameters:
	G (nx.DiGraph): The directed graph representing the causal structure.
	X (list): Set of treatment variables.
	Y (list): Set of outcome variables.

	Returns:
	list: The set of nodes suitable for adjustment.
	'''
	if check_adjustment_criterion(G,X,Y,[]):
		return set([])
	Z = construct_adjustment_set(G, X, Y)
	reacheable_Y = graph.find_reacheable_set(G, Y, Z, Z)
	Z1 = list(set(Z).intersection(set(reacheable_Y)))
	reacheable_X = graph.find_reacheable_set(G, X, Z, reacheable_Y)
	Z2 = list(set(Z1).intersection(set(reacheable_X)))
	return Z2 

