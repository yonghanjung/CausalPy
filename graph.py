import networkx as nx
import matplotlib.pyplot as plt
import random
import itertools
import copy
import adjustment
from collections import deque


def visualize(graph):
	''' Visualize the causal graph (in a form of nx.DiGraph) with colored nodes for treatments and outcomes. '''
	color_map = []
	for node in graph:
		if node.startswith('T') or node.startswith('X'):
			color_map.append('blue')  # Treatment nodes colored blue
		elif node.startswith('O') or node.startswith('Y'):
			color_map.append('red')  # Outcome nodes colored red
		elif node.startswith('U'):
			color_map.append('gray')  # Unobserved nodes colored gray
		else:
			color_map.append('lightblue')  # Other nodes
	
	pos = nx.get_node_attributes(graph, 'pos')
	if not pos:
		pos = nx.spring_layout(graph)

	nx.draw(graph, pos, with_labels=True, node_color=color_map, font_weight='bold', arrows=True)
	plt.title("Graph")
	plt.show(block=False)

def write_joint_distribution(variables):
	"""
	Writes the joint probability distribution symbolically.

	Parameters:
	variables (list): List of observed variables in topological order.

	Returns:
	str: Symbolic representation of the joint probability distribution.
	"""
	return f"P({' , '.join(variables)})"

def find_topological_order(G):
	"""
	Finds the topological order of the observed variables in the graph.

	Parameters:
	graph (nx.DiGraph): The projected causal graph.

	Returns:
	list: A list of observed variables in topological order.
	"""
	return [node for node in nx.topological_sort(G) if not node.startswith('U')]

def create_acyclic_graph(graph_dict, an_Y_graph_TF = False, Y = None, node_positions = None):
	'''
	Create a DAG from a dictionary representation, ensuring the graph is acyclic.

	** Parameters:
	- graph_dict (dict): A dictionary representing the graph, where keys are nodes, and values are lists of children nodes.
	- an_Y_graph_TF (bool): True if a returned graph is required to be an ancestor of a list Y. 
	- Y (list): A list of non-overlapted variable 
	- node_positions (dict): position of nodes

	** Returns:
	nx.DiGraph: The created directed graph (if acyclic).

	Raises:
	ValueError: If the graph is cyclic.
	'''

	# Preprocessing to make sure that Y doesn't have an overlapped variable 
	if Y != None:
		Y = list(set(Y))
	if an_Y_graph_TF == True and Y == None:
		raise ValueError("If an_Y_graph_TF = True, then Y must be a non-empty list")
	
	# Create a directed graph from the dictionary
	G = nx.DiGraph()
	for node, children in graph_dict.items():
		for child in children:
			G.add_edge(node, child)

	# Check for cycles in the graph
	if not nx.is_directed_acyclic_graph(G):
		raise ValueError("The graph is cyclic and cannot be created.")

	# Add positions to nodes if provided
	if node_positions:
		for node, position in node_positions.items():
			G.nodes[node]['pos'] = position

	return G if not an_Y_graph_TF else subgraphs(G, find_ancestor(G,Y))


def all_possible_orders_X(G,X):
	'''
	Generate multiple X in different order. 

	** Parameters 
	- G (nx.DiGraph): The directed acyclic graph.
	- X (list): a set of variable

	** Returns: 
	- X_list (list of list): A list of X with different orders. 
	'''
	def all_topological_orders(G, X):
		'''
		Generate all valid topological order of X 
		'''
		if not nx.is_directed_acyclic_graph(G):
			raise ValueError("The graph must be a directed acyclic graph (DAG)")
		observable_nodes = [node for node in G.nodes() if not node.startswith('U')]
		G_observables = G.subgraph(observable_nodes)
		for full_order in nx.all_topological_sorts(G_observables):
			# Filter the full topological order to only include nodes in X
			filtered_order = [node for node in full_order if node in X]
			yield filtered_order
	
	X_list = []
	for order in all_topological_orders(G, X):
		if order not in X_list:
			X_list.append(order)
	return X_list 


def subgraphs(G,C):
	'''
	Create a subgraph of G consisting of nodes in C and unobserved variables that are connected to nodes in C.

	Parameters:
	G (nx.DiGraph): The original causal graph.
	C (list): List of observed nodes for which the subgraph is to be created.

	Returns:
	nx.DiGraph: G(C)
	'''    

	subgraph_nodes = copy.copy(C)
	for node in list(C): 
		for neighbor in G.predecessors(node):
			if neighbor.startswith('U'):
				subgraph_nodes.append(neighbor)
		for neighbor in G.successors(node):
			if neighbor.startswith('U'):
				subgraph_nodes.append(neighbor)
	return G.subgraph(subgraph_nodes)


def list_all_c_components(G):
	'''
	List all c-components in the graph G, excluding unobserved variables.

	** Parameters:
	G (nx.DiGraph): The graph in which c-components are to be found.

	** Returns:
	list of list: A list where each element is a set of variables forming a c-component, excluding unobserved variables.
	'''

	def is_connected_through_latent(v1, v2):
		'''
		Checks if v1 <- U -> v2 for some unobservable U 

		Parameters:
		v1, v2: The observed variables to be checked for indirect connection via latent variables.

		Returns:
		bool: True if there is an indirect connection via a latent variable, False otherwise.
		'''
		for u in G.predecessors(v1):
			if u.startswith('U') and v2 in G.successors(u):
				return True
		return False

	# Assuming unobserved variables start with 'U'
	unobserved_vars = [n for n in G.nodes() if n.startswith('U')]
	observed_vars = [n for n in G.nodes() if not n.startswith('U')]
	modified_graph = G.copy()

	c_components = []
	visited = set()

	# Iterate through each observed variable in C to find connected components
	for v in observed_vars:
		if v not in visited:
			# Use BFS to find all observed variables connected through latent variables
			component = set()
			queue = [v]
			while queue:
				current = queue.pop(0)
				if current not in visited:
					visited.add(current)
					component.add(current)
					for neighbor in observed_vars:
						if neighbor != current and is_connected_through_latent(current, neighbor):
							queue.append(neighbor)

			# Filter out latent variables and add component if not empty
			observed_component = {node for node in component if not node.startswith('U')}
			if observed_component:
				c_components.append(list(observed_component))
	return c_components


def find_c_components(G, C):
	'''
	Identifies an union of c-components that contains C 

	Parameters:
	G (nx.DiGraph): The original causal graph.
	C (list): List of observed variables in the graph.

	Returns:
	list of list: Each set represents a c-component, a subset of observed variables connected through latent variables.
	'''

	c_component_G = list_all_c_components(G)
	set_C = set(C)

	combined_components = set()
	for element in set_C:
		for component in c_component_G:
			if element in component:
				combined_components = combined_components.union(component)
				break 

	return list(combined_components)


def find_parents(G, C):
	'''
	Identify the parents of nodes in C in the directed graph G.

	Parameters:
	G (nx.DiGraph): The directed graph.
	C (list): Nodes whose parents are to be found.

	Returns:
	set: A set of parent nodes of the nodes in C.
	'''
	parents = set()
	for node in C:
		if node in G:
			parents.update(G.predecessors(node))
	# Filter out latent variables and add component if not empty
	parents = {node for node in parents if not node.startswith('U') and node not in C}
	return list(parents)


def find_descendant(G, nodes):
	'''
	Find all observable descendants of a given set of nodes in a directed graph, excluding unobservables.

	Parameters:
	G (nx.DiGraph): The directed graph.
	nodes (list): Nodes whose observable descendants are to be found.

	Returns:
	set: A set of all observable descendants of the given nodes.
	'''
	descendants = set()
	for node in nodes:
		if not node.startswith('U'):
			descendants.add(node)
			descendants.update({n for n in nx.descendants(G, node) if not n.startswith('U')})
	return list(descendants)

def find_ancestor(G, nodes):
	'''
	Find all observable ancestors of a given set of nodes in a directed graph, excluding unobservables.

	Parameters:
	G (nx.DiGraph): The directed graph.
	nodes (list): Nodes whose observable ancestors are to be found.

	Returns:
	set: A set of all observable ancestors of the given nodes.
	'''
	ancestors = set()
	for node in nodes:
		if not node.startswith('U'):
			ancestors.add(node)
			ancestors.update({n for n in nx.ancestors(G, node) if not n.startswith('U')})
	return list(ancestors)

def subgraph_ancestor_Y(G,Y):
	'''
	Return G(AN(Y)) from G 
	'''
	return subgraphs(G, find_ancestor(G,Y))

def is_d_separated(G, X, Y, Z):
	return nx.d_separated(G, set(X), set(Y), set(Z))

def G_cut_incoming_edges(G, X):
	'''
	Return G_{bar{X}}. 

	Parameters:
	G (nx.DiGraph): The original directed graph.
	X (list): Nodes whose incoming edges are to be removed.

	Returns:
	nx.DiGraph: The modified graph with incoming edges to nodes in X removed.
	'''
	# Create a copy of the graph to avoid modifying the original graph
	G_modified = G.copy()

	# Iterate through each node in X and remove its incoming edges
	for node in X:
		if node in G_modified:
			# Get all incoming edges to the node
			incoming_edges = list(G_modified.in_edges(node))
			# Remove these edges
			G_modified.remove_edges_from(incoming_edges)

	return G_modified

def G_cut_outgoing_edges(G, X):
	'''
	Return G_{underline{X}}. 

	Parameters:
	G (nx.DiGraph): The original directed graph.
	X (list or set): Nodes whose outgoing edges are to be removed.

	Returns:
	nx.DiGraph: The modified graph with outgoing edges from nodes in X removed.
	'''
	# Create a copy of the graph to avoid modifying the original graph
	G_modified = G.copy()

	# Iterate through each node in X and remove its outgoing edges
	for node in X:
		if node in G_modified:
			# Get all outgoing edges from the node
			outgoing_edges = list(G_modified.out_edges(node))
			# Remove these edges
			G_modified.remove_edges_from(outgoing_edges)

	return G_modified


def generate_random_graph(num_observables, num_unobservables, num_treatments, num_outcomes, sparcity_constant = 0.25):
	'''
	Generate a random acyclic graph with specified numbers of observables, unobservables, treatments, and outcomes.

	Parameters:
	num_observables (int): Number of observable nodes.
	num_unobservables (int): Number of unobservable nodes.
	num_treatments (int): Number of treatment variables.
	num_outcomes (int): Number of outcome variables.

	Returns:
	tuple: Graph dictionary, node positions, lists X and Y.
	'''

	# Create observable nodes
	treatments = [f'X{i+1}' for i in range(num_treatments)]
	outcomes = [f'Y{i+1}' for i in range(num_outcomes)]
	other_observables = [f'V{i+1}' for i in range(num_observables - num_treatments - num_outcomes)]

	all_observables = treatments + outcomes + other_observables
	is_acyclic = False

	while not is_acyclic:
		G = nx.DiGraph()

		# Add unobservable edges
		unobservable_edges = set()
		for i in range(num_unobservables):
			obs_pair = random.sample(all_observables, 2)
			unobservable = f'U_{"_".join(obs_pair)}'
			for child in obs_pair:
				unobservable_edges.add((unobservable, child))

		G.add_edges_from(unobservable_edges)

		# Optional: Add additional edges between observables
		additional_edges = [(a, b) for a in all_observables for b in all_observables if a != b]
		random.shuffle(additional_edges)

		additional_edges = random.sample(additional_edges, round(len(additional_edges)*sparcity_constant))
		for edge in additional_edges:
			G.add_edge(*edge)
			if not nx.is_directed_acyclic_graph(G):
				G.remove_edge(*edge)

		if nx.is_directed_acyclic_graph(G):
			if set(outcomes).issubset(set(G.nodes)):
				break 
			else:
				continue


	# Generate node positions for visualization
	node_positions = {node: (random.uniform(0, 100), random.uniform(0, 100)) for node in G.nodes()}

	# Convert graph to dictionary format
	graph_dict = {node: list(G.successors(node)) for node in G.nodes()}

	return [graph_dict, node_positions, treatments, outcomes]


def check_inducing_paths(G, nodes1, nodes2, S, L):
	'''
	Check if there is an inducing path between node1 and node2 in G, given conditioning set S and the marginalized set L.

	Parameters:
	G (nx.DiGraph): The original directed graph.
	nodes1 (list)
	nodes2 (list)
	S (list)
	L (list)
	'''
	# def check_inducing_path(path):
	# 	''' Check if the path is valid considering colliders and non-colliders. '''
	# 	for i in range(1, len(path) - 1):
	# 		node = path[i]
	# 		is_collider = G.has_edge(path[i-1], node) and G.has_edge(path[i+1], node)
	# 		if not is_collider: # If a node is not a collder, 
	# 			if node not in L: #... it must be in L 
	# 				return False 
	# 		else: # If a node is a collider, 
	# 			if node not in list(find_ancestor(G,S) + S):
	# 				return False 
	# 	return True 

	def is_inducing_node(node, prev_node, next_node):
		is_collider = G.has_edge(prev_node, node) and G.has_edge(next_node, node)
		if is_collider:
			return node in list(find_ancestor(G,S) + S)
		else:
			return node in L

	# Convert G to undirected for path finding
	undirected_G = G.to_undirected()

	# Check paths between each pair of start and end nodes
	for start in nodes1:
		for end in nodes2:
			if start != end:
				# Perform a BFS from the start node to find paths to the end node
				queue = [(start, [start])]
				while queue:
					(current_node, path) = queue.pop(0)
					if current_node == end:
						if all(is_inducing_node(path[i], path[i-1], path[i+1]) for i in range(1, len(path) - 1)):
							return True
					for neighbor in undirected_G[current_node]:
						if neighbor not in path:
							queue.append((neighbor, path + [neighbor]))


	# for start in nodes1:
	# 	for end in nodes2:
	# 		if start != end:
	# 			all_paths = nx.all_simple_paths(undirected_G, source=start, target=end)
	# 			for path in all_paths:
	# 				if is_valid_path(path):
	# 					return True

	return False


def is_inducing_path_with_unmeasured(G, nodes1, nodes2):
	# Define L as the set of unmeasured (unobserved) variables in the graph
	# Assuming unmeasured variables are not in a specific list or have a specific naming pattern
	# Adjust this line based on how unmeasured variables are represented in your graph
	L = {node for node in G.nodes() if is_unmeasured(node)}

	# Call the existing is_inducing_path function with S as an empty set
	if check_inducing_paths(G, nodes1, nodes2, S=[], L=L):
		return True

	return False

def is_unmeasured(node):
	# Define how to determine if a node is unmeasured
	# This is a placeholder function, implement according to your graph's characteristics
	return node.startswith('U')

def find_variables_no_inducing_path(G, nodes):
	'''
	Given a set of variables "nodes", find all other variables that has no inducing path to "nodes". 

	Parameters:
	G (nx.DiGraph): The original directed graph.
	nodes (list)
	'''

	V = [node for node in nx.topological_sort(G) if not node.startswith('U')]
	V_minus_nodes = list(set(V) - set(nodes))
	list_variables = []

	for Vi in V_minus_nodes:
		# print(Vi, [Vi])
		if not is_inducing_path_with_unmeasured(G,[Vi],nodes):
			list_variables.append(Vi)
	return list_variables


def find_reacheable_set(G, X, A, Z):
	'''
	Find the subset of A that is reachable from X conditioned on Z.
	'''

	X_set = set(X)
	A_set = set(A)
	Z_set = set(Z)
	U_set = set([n for n in G.nodes() if n.startswith('U')])
	ancestors_of_Z = find_ancestor(G, Z)
	
	def is_valid_path(path):
		for i in range(1, len(path) - 1):
			node = path[i]
			is_collider = G.has_edge(path[i-1], node) and G.has_edge(path[i+1], node)
			if is_collider:
				if node not in list(Z_set.union(find_ancestor(G,Z))):
					return False
			else:
				if node in Z_set:
					return False
		return True

	# Convert G to undirected for path finding
	undirected_G = G.to_undirected()

	closure = set()
	
	for x in X:
		# BFS queue initialized with the starting node and the path
		queue = deque([(x, [x])])
		visited = set()
		
		while queue:
			current_node, path = queue.popleft()
			
			# If current node is in A_set and not visited, add to closure
			if current_node in A_set.union(set([x])) and current_node not in visited:
				closure.add(current_node)
				visited.add(current_node)
			
			for neighbor in undirected_G.neighbors(current_node):
				if neighbor not in path and neighbor in A_set.union(set([x])).union(U_set): 
					new_path = path + [neighbor]
					if is_valid_path(new_path):
						queue.append((neighbor, new_path))
	
	return list(closure.union(X_set))

def bayes_ball_search(G, X, Z):
	'''
	Perform Bayes-Ball Search to find the set of vertices d-connected to X given Z in G.
	
	G : NetworkX DiGraph (Directed Acyclic Graph)
	X : list of source nodes
	Z : list of conditioning nodes
	'''
	X = set(X)
	Z = set(Z)
	V = set(G.nodes())
	visited = {v: {'inc': False, 'out': False} for v in V}
	
	def visit(G, V, edgetype):
		visited[V][edgetype] = True
		if V not in Z:
			for W in G.successors(V):  # Children of V
				if not visited[W]['inc']:
					visit(G, W, 'inc')
		if (edgetype == 'inc' and V in Z) or (edgetype == 'out' and V not in Z):
			for W in G.predecessors(V):  # Parents of V
				if not visited[W]['out']:
					visit(G, W, 'out')
	
	for X_node in X:
		if not visited[X_node]['out']:
			visit(G, X_node, 'out')
	
	return list({v for v in V if visited[v]['inc'] or visited[v]['out']})

def graph_dict_to_fusion_graph(graph_dict):
	'''
	Output the graph_dict as Fusion 
	'''
	nodes = set()
	edges = []

	# Extract nodes and edges
	for key, values in graph_dict.items():
		if key.startswith("U_"):  # Handle latent variables
			if len(values) == 2:  # Assuming latent variables connect exactly two nodes
				edge_weight = random.uniform(-1, 1)  # Placeholder weight
				edges.append(f"{values[0]} -- {values[1]} {edge_weight:.2f}")
		else:
			nodes.add(key)
			for value in values:
				edge_type = "->"
				edges.append(f"{key} {edge_type} {value}")
				if not value.startswith("U_"):
					nodes.add(value)

	# Generate random positions for nodes
	positions = {node: (random.randint(-200, 200), random.randint(-200, 200)) for node in nodes}

	# Format the output
	fusion_graph = "<NODES>\n"
	fusion_graph += "\n".join(f"{node}  {pos[0]},{pos[1]}" for node, pos in positions.items())
	fusion_graph += "\n\n<EDGES>\n"
	fusion_graph += "\n".join(edges)

	return fusion_graph


