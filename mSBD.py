import networkx as nx
import graph
import adjustment    

def partition_Y(G, X, Y):
	'''
	Partition the set Y based on the descendants of nodes in X.

	Parameters:
	G (nx.DiGraph): The directed graph representing the causal structure.
	X (list): List of treatment variables, topologically ordered.
	Y (list): List of outcome variables.

	Returns:
	dict: A dictionary where keys are Y0, Y1, ..., Ym and values are the corresponding partitions of Y.
	'''

	# Ensure X is topologically sorted
	X_topo_sorted = list(nx.topological_sort(G))
	X = [x for x in X_topo_sorted if x in X]

	# Initialize the partition dictionary
	partition = dict()
	
	# Calculate Y0
	all_descendants_of_X = graph.find_descendant(G, X)
	partition['Y0'] = list(set(Y) - set(all_descendants_of_X))

	# Calculate Yi for i = 1 to m
	for i, Xi in enumerate(X):
		so_far_Y = list()
		for j in range(i+1):
			so_far_Y += partition[f'Y{i}']
		remaining_Y = list(set(Y) - set(so_far_Y))
		descendants_of_Xi = graph.find_descendant(G, [Xi])
		if i < len(X) - 1:
			descendants_of_next_X = graph.find_descendant(G, X[i+1:])
			partition[f'Y{i+1}'] = list(set(remaining_Y).intersection(set(descendants_of_Xi) - set(descendants_of_next_X)))
		else:
			partition[f'Y{i+1}'] = list(set(remaining_Y).intersection(set(descendants_of_Xi)))
	return partition

def check_mSBD_criterion_fixed_order(G, X, Y, Z):
	'''
	Check mSBD criterion with the fixed topological order of X 

	Parameters:
	G (nx.DiGraph): The directed graph representing the causal structure.
	X (list): List of treatment variables, topologically ordered.
	Y (list): List of outcome variables.
	Z (dict): Dictionary in the form of, e.g., {'Z1': ['Z1','Z2'], 'Z2': ['Z3', 'Z4']}.

	Returns:
	True if Z satisfies the mSBD criterion relative to X and Y. False otherwise. 
	'''

	Y_partitions = partition_Y(G, X, Y)
	for i, Xi in enumerate(X):
		# Check non-descendant condition for Zi
		# Zi = Z_partitions.get(f'Z{i+1}', set())
		Zi = Z.get(f'Z{i+1}', [])
		# "False" if Zi is included in De(Xi, Xi+1, ... )
		if any(node in graph.find_descendant(G, X[i:]) for node in Zi):
			return False

		# Prepare graph G_oi for d-separation condition
		G_oi = graph.G_cut_incoming_edges(graph.G_cut_outgoing_edges(G, [Xi]), X[i+1:])
		
		# Construct conditioning set
		past_Z = list(set().union(*[Z[f'Z{j}'] for j in range(1, i+1)])) # i+1 is a current index. By the range, range(, i+1) is right before the current
		past_Y = list(set().union(*[Y_partitions[f'Y{j}'] for j in range(1, i+1)]))
		past_X = list(set(X) - set(X[i:]))
		history_i = (past_Z + past_Y + past_X)
		future_Y = list(set(Y) - set(past_Y))
		conditioning_set = history_i + Zi

		if not graph.is_d_separated(G_oi, [Xi], future_Y, conditioning_set):
			return False

	return True

def construct_mSBD_Z(G, X, Y):
	'''
	Construct the candidate Z for checking mSBD

	Parameters:
	G (nx.DiGraph): The directed graph representing the causal structure.
	X (list): List of treatment variables.
	Y (list): List of outcome variables.

	Returns:
	dict: A dictionary where keys are indices and values are the Zi sets.
	'''
	X = list(X)
	Y_partitions = partition_Y(G, X, Y)
	Z = dict()
	m = max([int(key[1:]) for key in Y_partitions.keys()])
	for i, Xi in enumerate(X):
		# Construct Gi by removing incoming edges to {X_{i+1}, X_{i+2}, ...}
		# Gi = graph.G_cut_incoming_edges(G, remaining_X)
		G_oi = graph.G_cut_incoming_edges(graph.G_cut_outgoing_edges(G, [Xi]), X[i+1:])

		# Compute future_Y as {Yi, Yi+1, ...}
		past_Z = list(set().union(*[Z[f'Z{j}'] for j in range(1, i+1)])) # i+1 is a current index. By the range, range(, i+1) is right before the current
		past_Y = list(set().union(*[Y_partitions[f'Y{j}'] for j in range(1, i+1)]))
		past_X = list(set(X) - set(X[i:]))
		history_i = (past_Z + past_Y + past_X)

		De_X_i = graph.find_descendant(G, X[i:])
		Forbidden_list = list( set(X) | set(Y) | set(history_i) | set(De_X_i) )

		future_Y = list(set(Y) - set(past_Y))
		Zi = list( set( graph.find_ancestor(G_oi,[Xi] + future_Y + history_i ) ) - set(Forbidden_list) )
		# if Zi:
		Z[f'Z{i+1}'] = Zi

	return Z

def constructive_mSBD_criterion_fixed_order(G, X, Y):
	"""
	Check if Z satisfies the modified Sequential Back-Door (mSBD) criterion in graph G.

	Parameters:
	G (nx.DiGraph): The directed graph.
	X (list): List of treatment variables.
	Y (list): List of outcome variables.

	Returns:
	bool: True if P(Y | do(x)) can be written as the mSBD 
	"""

	return check_mSBD_criterion_fixed_order(G,X,Y,construct_mSBD_Z(G,X,Y))


def constructive_mSBD_criterion(G, X, Y):
	"""
	Check if Z satisfies the modified Sequential Back-Door (mSBD) criterion in graph G.

	Parameters:
	G (nx.DiGraph): The directed graph.
	X (list): List of treatment variables.
	Y (list): List of outcome variables.

	Returns:
	bool: True if P(Y | do(x)) can be written as the mSBD 
	"""

	X_list = graph.all_possible_orders_X(G,X)
	for X_order in X_list:
		if constructive_mSBD_criterion_fixed_order(G,X_order,Y):
			return True 
	return False


def check_mSBD_with_results(G,X,Y):
	"""
	Check if P(Y | do(=X)) can be represented as an mSBD, and if so, provide the partitioned X,Z,Y

	Parameters:
	G (nx.DiGraph): The directed graph.
	X (list): List of treatment variables.
	Y (list): List of outcome variables.

	Returns:
	dict: if mSBD admissible, the dictionaries for X, Z, Y. Otherwise, raise an error "not mSBD admissible"
	"""
	if not constructive_mSBD_criterion(G,X,Y): 
		raise ValueError("Not mSBD Admissible")

	X_list = graph.all_possible_orders_X(G,X)
	for X_order in X_list:
		if constructive_mSBD_criterion(G,X_order,Y):
			break 

	m = len(X_order)

	dict_X = {}
	for idx in range(len(X_order)):
		dict_X[f"X{idx+1}"] = {X_order[idx]}
		
	dict_Y = partition_Y(G, X_order, Y)

	dict_Z = construct_mSBD_Z(G, X_order, Y)

	for i in reversed(range(m)):
		idx = i + 1
		if len(dict_Y[f"Y{idx}"]) > 0:
			break 
		else:
			del dict_Y[f"Y{idx}"]
			del dict_Z[f"Z{idx}"]
			del dict_X[f"X{idx}"]

	return [dict_X, dict_Z, dict_Y]


def check_SAC_criterion_fixed_order(G, X, Y, Z):
	'''
	Check SAC criterion with the fixed topological order of X 

	Parameters:
	G (nx.DiGraph): The directed graph representing the causal structure.
	X (list): List of treatment variables, topologically ordered.
	Y (list): List of outcome variables.
	Z (dict): Dictionary in the form of, e.g., {'Z1': ['Z1','Z2'], 'Z2': ['Z3', 'Z4']}.

	Returns:
	True if Z satisfies the mSBD criterion relative to X and Y. False otherwise. 
	'''
	Y_partitions = partition_Y(G, X, Y)
	for i, Xi in enumerate(X):
		past_Z = list(set().union(*[Z[f'Z{j}'] for j in range(1, i+1)])) # i+1 is a current index. By the range, range(, i+1) is right before the current
		past_Y = list(set().union(*[Y_partitions[f'Y{j}'] for j in range(0, i+1)]))
		past_X = list(set(X) - set(X[i:]))
		history_i = (past_Z + past_Y + past_X)
		future_Y = list(set(Y) - set(past_Y))
		Zi = Z.get(f'Z{i+1}', [])
		# "False" if Zi is included in De(Xi+1, ... )
		if any(node in graph.find_descendant(G, X[i+1:]) for node in Zi):
			return False

		if any(node in adjustment.descedent_proper_causal_path(G,[Xi], future_Y) for node in Zi):
			return False
		
		G_psbd_i = adjustment.proper_backdoor_graph(graph.G_cut_incoming_edges(G, X[i+1:]), [Xi], future_Y)
		conditioning_set = history_i + Zi

		# Check d-separation condition in G_oi
		if not graph.is_d_separated(G_psbd_i, [Xi], future_Y, conditioning_set):
			return False

	return True


def construct_SAC_Z(G, X, Y):
	'''
	Construct the candidate Z for checking mSBD

	Parameters:
	G (nx.DiGraph): The directed graph representing the causal structure.
	X (list): List of treatment variables.
	Y (list): List of outcome variables.

	Returns:
	dict: A dictionary where keys are indices and values are the Zi sets.
	'''
	X = list(X)
	Y_partitions = partition_Y(G, X, Y)
	Z = dict()
	m = max([int(key[1:]) for key in Y_partitions.keys()])
	for i, Xi in enumerate(X):
		past_Z = list(set().union(*[Z[f'Z{j}'] for j in range(1, i+1)])) # i+1 is a current index. By the range, range(, i+1) is right before the current
		past_Y = list(set().union(*[Y_partitions[f'Y{j}'] for j in range(1, i+1)]))
		past_X = list(set(X) - set(X[i:]))
		history_i = (past_Z + past_Y + past_X)
		future_Y = list(set(Y) - set(past_Y))

		G_psbd_i = adjustment.proper_backdoor_graph(graph.G_cut_incoming_edges(G, X[i+1:]), [Xi], future_Y)

		# Compute future_Y as {Yi, Yi+1, ...}
		past_Z = list(set().union(*[Z[f'Z{j}'] for j in range(1, i+1)])) # i+1 is a current index. By the range, range(, i+1) is right before the current
		past_Y = list(set().union(*[Y_partitions[f'Y{j}'] for j in range(1, i+1)]))
		past_X = list(set(X) - set(X[i:]))
		history_i = (past_Z + past_Y + past_X)
		dpcp_i = adjustment.descedent_proper_causal_path(G, [Xi], future_Y)
		De_X_i1 = graph.find_descendant(G, X[i+1:])

		Forbidden_list = list( set(X) | set(Y) | set(history_i) | set(De_X_i1)  | set(dpcp_i))
		Zi = list( set( graph.find_ancestor(G_psbd_i,[Xi] + future_Y + history_i ) ) - set(Forbidden_list) )
		# if Zi:
		Z[f'Z{i+1}'] = Zi
	return Z

def construct_minimum_SAC_Z(G,X,Y):
	'''
	Construct the minimal candidate Z for checking mSBD

	Parameters:
	G (nx.DiGraph): The directed graph representing the causal structure.
	X (list): List of treatment variables.
	Y (list): List of outcome variables.

	Returns:
	dict: A dictionary where keys are indices and values are the Zi sets.
	'''
	Y_partitions = partition_Y(G, X, Y)
	Z = construct_SAC_Z(G,X,Y)
	m = max([int(key[1:]) for key in Y_partitions.keys()])
	Zmin = dict()
	for i, Xi in enumerate(X):
		past_Z = list(set().union(*[Zmin[f'Z{j}'] for j in range(1, i+1)])) # i+1 is a current index. By the range, range(, i+1) is right before the current
		past_Y = list(set().union(*[Y_partitions[f'Y{j}'] for j in range(1, i+1)]))
		past_X = list(set(X) - set(X[i:]))
		history_i = (past_Z + past_Y + past_X)
		future_Y = list(set(Y) - set(past_Y))

		G_psbd_i = adjustment.proper_backdoor_graph(graph.G_cut_incoming_edges(G, X[i+1:]), [Xi], future_Y)

		Zi = Z[f'Z{i+1}']
		reacheable_Y = graph.find_reacheable_set(G_psbd_i, future_Y, Zi + history_i , Zi + history_i)
		Zi_1 = list(set(Zi).intersection(set(reacheable_Y)))
		reacheable_X = graph.find_reacheable_set(G_psbd_i, [Xi], Zi+ history_i, Zi_1 + history_i)
		Zi_min = list(set(Zi_1).intersection(set(reacheable_X)))

		Zmin[f'Z{i+1}'] = Zi_min
	return Zmin


def constructive_SAC_criterion_fixed_order(G, X, Y):
	"""
	Check if Z satisfies the modified Sequential Back-Door (mSBD) criterion in graph G.

	Parameters:
	G (nx.DiGraph): The directed graph.
	X (list): List of treatment variables.
	Y (list): List of outcome variables.

	Returns:
	bool: True if P(Y | do(x)) can be written as the mSBD 
	"""

	return check_SAC_criterion_fixed_order(G,X,Y,construct_SAC_Z(G,X,Y))


def constructive_SAC_criterion(G, X, Y):
	"""
	Check if Z satisfies the modified Sequential Back-Door (mSBD) criterion in graph G.

	Parameters:
	G (nx.DiGraph): The directed graph.
	X (list): List of treatment variables.
	Y (list): List of outcome variables.

	Returns:
	bool: True if P(Y | do(x)) can be written as the mSBD 
	"""

	X_list = graph.all_possible_orders_X(G,X)
	for X_order in X_list:
		if constructive_SAC_criterion_fixed_order(G,X_order,Y):
			return True 
	return False


def check_SAC_with_results(G,X,Y, minimum = False):
	"""
	Check if P(Y | do(=X)) can be represented as an sequential admissible, and if so, provide the partitioned X,Z,Y

	Parameters:
	G (nx.DiGraph): The directed graph.
	X (list): List of treatment variables.
	Y (list): List of outcome variables.

	Returns:
	dict: if sequential admissible admissible, the dictionaries for X, Z, Y. Otherwise, raise an error "not sequential admissible admissible"
	"""
	if not constructive_SAC_criterion(G,X,Y): 
		raise ValueError("Not Sequential Covariate Admissible")

	X_list = graph.all_possible_orders_X(G,X)
	for X_order in X_list:
		if constructive_SAC_criterion(G,X_order,Y):
			break 

	m = len(X_order)

	dict_X = dict()
	for idx in range(len(X_order)):
		dict_X[f"X{idx+1}"] = [X_order[idx]]
		
	dict_Y = partition_Y(G, X_order, Y)

	if minimum:
		dict_Z = construct_minimum_SAC_Z(G, X_order, Y)
	else:
		dict_Z = construct_SAC_Z(G, X_order, Y)

	for i in reversed(range(m)):
		idx = i + 1
		if len(dict_Y[f"Y{idx}"]) > 0:
			break 
		else:
			del dict_Y[f"Y{idx}"]
			del dict_Z[f"Z{idx}"]
			del dict_X[f"X{idx}"]

	return [dict_X, dict_Z, dict_Y]



def mSBD_estimand(G, X, Y, latex = False, minimum=False):
	"""
	Provide the estimand for the mSBD adjustment

	Parameters:
	G (nx.DiGraph): The directed graph.
	X (list): List of treatment variables.
	Y (list): List of outcome variables. 
	latex (bool): True if the estimand is in the latex syntax. 
	minimum (bool): If Z needs to be the minimum sequential covariate 

	Returns:
	str: mSBD estimand
	"""

	if adjustment.check_admissibility(G,X,Y):
		if minimum:
			Z = adjustment.construct_minimum_adjustment_set(G, X, Y)
		else:
			Z = adjustment.construct_adjustment_set(G, X, Y)
		return adjustment.adjustment_estimand(X,Y,Z,latex)

	dict_X, dict_Z, dict_Y = check_SAC_with_results(G, X, Y, minimum)
	m = len(dict_X)  # Assuming all dictionaries have the same length

	dict_X["X0"] = list()
	dict_Z["Z0"] = list()

	dict_H = {f"H0": dict_X[f"X{0}"] + dict_Y[f"Y{0}"] + dict_Z[f"Z{0}"]}
	for i in range(1,m):
		dict_H[f"H{i}"] =  dict_X[f"X{i}"] + dict_Y[f"Y{i}"] + dict_Z[f"Z{i}"] + dict_H[f"H{i-1}"]

	term_list = []
	for i in range(m):
		idx = i + 1
		Xi_1 = dict_X.get(f"X{i}", list())
		Yi_1 = dict_Y.get(f"Y{i}", list())
		Zi = dict_Z.get(f"Z{idx}", list())
		Zi_1 = dict_Z.get(f"Z{i}", list())

		Yi_1_Zi = Yi_1 + Zi
		given_terms = dict_H.get(f"H{i-1}", list()) + Xi_1 + Zi_1

		if len(Yi_1_Zi) > 0:
			if not latex:
				term = f"P({', '.join(Yi_1_Zi)}" + (f" | {', '.join(given_terms)}" if given_terms else "") + ")"
			else:
				term = f"P({', '.join(Yi_1_Zi)}" + (f" \\mid {', '.join(given_terms)}" if given_terms else "") + ")"
		else: 
			continue
		term_list.append(term)

	Ym = dict_Y[f"Y{m}"]
	given_term_m = dict_H[f"H{m-1}"] + dict_X[f"X{m}"] + dict_Z[f"Z{m}"] 
	if len(Ym) > 0:
		if not latex: 
			term_list.append(f"P({', '.join(Ym)}" + (f" | {', '.join(given_term_m)}" if given_term_m else "") + ")")
		else:
			term_list.append(f"P({', '.join(Ym)}" + (f" \\mid {', '.join(given_term_m)}" if given_term_m else "") + ")")

	summands = {z.lower() for values in dict_Z.values() for z in values}
	summands_str = ', '.join(summands)
	term_list_expression = ' '.join(reversed(term_list))

	if len(summands) > 0:
		if not latex:
			final_estimand = f"\u03A3_{{{summands_str}}} {term_list_expression}"
		else: 
			final_estimand = f"\\sum_{{{summands_str}}} {term_list_expression}"
	else:
		final_estimand = f"{term_list_expression}"

	return final_estimand
