import numpy as np 
from pyvis.network import Network # Added import
import networkx as nx
import matplotlib.pyplot as plt
import random
import copy
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

def visualize_interactive(graph, filename="interactive_graph.html", edge_curvature=0):
    """
    Creates an interactive HTML visualization of the causal graph using pyvis.
    Nodes can be dragged and moved independently after disabling physics in the UI.

    ** Parameters:
    - graph (nx.DiGraph): The graph to visualize.
    - filename (str): The name of the HTML file to save.
    """
    if not graph.nodes():
        print("Graph is empty. Nothing to visualize.")
        return

    # Create a pyvis network object
    net = Network(height="750px", width="100%", directed=True, notebook=True, cdn_resources='in_line')

    # Load the networkx graph into pyvis
    net.from_nx(graph)

    # Customize node colors based on the logic in the original visualize function
    for node in net.nodes:
        node_id = str(node["id"])
        if node_id.startswith('T') or node_id.startswith('X'):
            node["color"] = 'blue'
        elif node_id.startswith('O') or node_id.startswith('Y'):
            node["color"] = 'red'
        elif node_id.startswith('U'):
            node["color"] = 'gray'
        else:
            node["color"] = 'lightblue'
            
    # === NEW: Set edge curvature ===
    # Iterate through the edges to set their smoothness property for curvature
    for edge in net.edges:
        edge['smooth'] = {'type': 'curvedCW', 'roundness': edge_curvature}

    # Add a UI to the HTML file to control physics settings.
    net.show_buttons(filter_=['physics'])

    try:
        net.show(filename)
        print(f"Interactive graph saved to {filename}")
        print("Open this file in a browser. To move nodes freely, uncheck the 'enabled' box in the 'physics' menu.")
    except Exception as e:
        print(f"An error occurred while generating the interactive graph: {e}")

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
	# if Y != None:
	# 	Y = list(set(Y))
	# if an_Y_graph_TF == True and Y == None:
	# 	raise ValueError("If an_Y_graph_TF = True, then Y must be a non-empty list")
	
	# Create a directed graph from the dictionary
	G = nx.DiGraph()
	
	# FIX: Add all nodes to the graph first.
	# The node_positions dictionary reliably contains all nodes that should exist.
	if node_positions:
		G.add_nodes_from(node_positions.keys())
 
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

def find_children(G, C):
	"""
	Identify the children of nodes in C in the directed graph G.

	Parameters:
	G (nx.DiGraph): The directed graph.
	C (list): Nodes whose children are to be found.

	Returns:
	set: A set of children nodes of the nodes in C.
	"""
	children = set()
	for node in C:
		if node in G:
			children.update(G.successors(node))
	# Filter out latent variables and add component if not empty
	children = {node for node in children if not node.startswith('U') and node not in C}
	return list(children)



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


def generate_random_graph(num_observables, num_unobservables, num_treatments,
                          num_outcomes, sparcity_constant=0.25, seednum=123):
    """
    Efficiently generates a random Acyclic Directed Mixed Graph (ADMG) by construction.

    This function first defines a random topological order for the observable nodes
    to guarantee the resulting directed graph is acyclic. It then adds unobserved
    confounders with canonical names to prevent duplicates like U_A_B and U_B_A.

    Parameters:
    num_observables (int): Total number of observable nodes.
    num_unobservables (int): Number of unique unobserved confounders to add.
    num_treatments (int): Number of treatment variables (X).
    num_outcomes (int): Number of outcome variables (Y).
    sparcity_constant (float): Used to determine the probability of an edge.
    seednum (int, optional): Random seed for reproducibility.

    Returns:
    list: A list containing [graph_dict, node_positions, treatments, outcomes].
    """
    if seednum is not None:
        random.seed(seednum)
        np.random.seed(seednum)

    # 1. Define all observable nodes
    treatments = [f'X{i+1}' for i in range(num_treatments)]
    outcomes = [f'Y{i+1}' for i in range(num_outcomes)]
    other_observables = [f'V{i+1}' for i in range(num_observables - num_treatments - num_outcomes)]
    all_observables = treatments + outcomes + other_observables

    # 2. Establish a random topological ordering to guarantee acyclicity
    random.shuffle(all_observables)
    
    G = nx.DiGraph()
    G.add_nodes_from(all_observables)

    edge_prob = min(sparcity_constant * 2, 1.0)

    # 3. Add directed edges based on the ordering (guarantees a DAG)
    for i, node_u in enumerate(all_observables):
        for j, node_v in enumerate(all_observables):
            if i < j:
                if random.random() < edge_prob:
                    G.add_edge(node_u, node_v)

    # 4. Add unobserved confounders (Corrected Logic)
    confounded_pairs = set()
    num_nodes = len(all_observables)
    max_possible_confounders = num_nodes * (num_nodes - 1) // 2

    # Ensure we don't request more confounders than possible unique pairs
    if num_unobservables > max_possible_confounders:
        print(f"Warning: Requested {num_unobservables} unobservables, but only {max_possible_confounders} unique pairs exist. Generating {max_possible_confounders}.")
        num_unobservables = max_possible_confounders

    # Use a while loop to find unique pairs to confound
    max_tries = num_unobservables * 20  # Safety break to prevent infinite loops
    tries = 0
    while len(confounded_pairs) < num_unobservables and tries < max_tries:
        tries += 1
        if num_nodes > 1:
            # Randomly select two observable nodes
            obs_pair = random.sample(all_observables, 2)
            
            # Create a canonical (sorted) tuple to represent the pair
            canonical_pair = tuple(sorted(obs_pair))
            
            # If this pair is already confounded, skip and try again
            if canonical_pair in confounded_pairs:
                continue
            
            # Store the new confounded pair
            confounded_pairs.add(canonical_pair)

            # Create a unique, canonical name for the unobserved node
            u_var = f'U_{canonical_pair[0]}_{canonical_pair[1]}'
            
            # Add the unobserved node and its edges
            G.add_node(u_var)
            G.add_edge(u_var, canonical_pair[0])
            G.add_edge(u_var, canonical_pair[1])

    if len(confounded_pairs) < num_unobservables:
        print(f"Warning: Could only generate {len(confounded_pairs)} unique confounders after {max_tries} attempts.")

    # 5. Generate final outputs
    node_positions = {node: (random.uniform(0, 100), random.uniform(0, 100)) for node in G.nodes()}
    graph_dict = {node: list(G.successors(node)) for node in G.nodes()}

    return [graph_dict, node_positions, treatments, outcomes]

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


def do_calculus_1(G, Y, Z, X, W):
	'''
	Check (Y indep Z | X, W)_{do(X)}

	Parameters:
	G (nx.DiGraph): The original directed graph.
	Y (list)
	Z (list)
	X (list)
	W (list)

	Returns:
	bool: True False
	'''
	return is_d_separated(G_cut_incoming_edges(G,X), Y, Z, W + X)

def do_calculus_2(G, Y, Z, X, W):
	'''
	Check (Y indep Z | X, W)_{do(X)}

	Parameters:
	G (nx.DiGraph): The original directed graph.
	Y (list)
	Z (list)
	X (list)
	W (list)

	Returns:
	bool: True False
	'''
	return is_d_separated(G_cut_outgoing_edges(G_cut_incoming_edges(G,X), Z), Y, Z, W + X)

def do_calculus_3(G, Y, Z, X, W):
	'''
	Check (Y indep Z | X, W)_{do(X)}

	Parameters:
	G (nx.DiGraph): The original directed graph.
	Y (list)
	Z (list)
	X (list)
	W (list)

	Returns:
	bool: True False
	'''
	Z_W = list( set(Z) - set(find_ancestor(G_cut_incoming_edges(G,X), W)) )
	return is_d_separated(G_cut_incoming_edges(G_cut_incoming_edges(G,X), Z_W), Y, Z, W + X)

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


def contiguous_series(G,W,topo_V = None):
	if topo_V == None:
		topo_V = find_topological_order(G)
	position = {node: i for i, node in enumerate(topo_V)}
	
	current_seq = []
	best_seq = []
	max_length = 0

	# Iterate through the elements in W
	for node in W:
		if node in position:
			# If the current sequence is empty or the current node's position is
			# exactly 1 greater than the previous node's position in the sequence,
			# then continue the sequence
			if not current_seq or position[node] == position[current_seq[-1]] + 1:
				current_seq.append(node)
			else:
				# If the current sequence is broken, check if it's the longest found so far
				if len(current_seq) > max_length:
					best_seq = current_seq
					max_length = len(current_seq)
				# Start a new sequence
				current_seq = [node]
	
	# Check the last sequence after the loop
	if len(current_seq) > max_length:
		best_seq = current_seq
	
	return list(tuple(best_seq))

def find_successors(G,W,topo_V = None):
	W_cont = contiguous_series(G,W)
	if set(W_cont) != set(W):
		raise(f"{W} is not contiguous")
	
	if topo_V == None:
		topo_V = find_topological_order(G)
	W = sorted(W, key=lambda x: topo_V.index(x))
	return(topo_V[topo_V.index(W[-1])+1:])

def find_predecessors(G,W,topo_V = None):
	if topo_V == None:
		topo_V = find_topological_order(G)
	W = sorted(W, key=lambda x: topo_V.index(x))
	return(topo_V[:topo_V.index(W[0])])

def expand_variables(variable_list, cluster_map):
    """Expands conceptual cluster nodes into their full list of column names."""
    if not variable_list or not cluster_map:
        return variable_list

    expanded_list = []
    for var in variable_list:
        # If var is a cluster name, extend the list with its columns
        if var in cluster_map:
            expanded_list.extend(cluster_map[var])
        # Otherwise, it's a regular variable
        else:
            expanded_list.append(var)
    return expanded_list

def build_cluster_map(conceptual_nodes, obs_data):
    """
    Automatically builds a cluster map dictionary by scanning DataFrame columns.

    This function uses a 'startswith' naming convention to find component columns
    for each conceptual node.

    Parameters:
      conceptual_nodes: A list of base variable names that might be clusters (e.g., ['C', 'Z']).
      obs_data: A pandas DataFrame.

    Returns:
      A dictionary mapping conceptual nodes to their component columns.
      e.g., {'C': ['C1', 'C2', 'C10'], 'Z': ['Z1', 'Z2']}
    """
    cluster_map = {}
    for node in conceptual_nodes:
        # Find all columns that start with the node name, but are not the node name itself
        components = [col for col in obs_data.columns if col.startswith(node) and col != node]
        
        if components:
            # Optional: a more robust way to sort numeric suffixes
            try:
                components.sort(key=lambda x: int(x[len(node):]))
            except ValueError:
                components.sort() # Fallback to alphabetical sort
            
            cluster_map[node] = components
            
    return cluster_map


def graph_to_graphdict(G: nx.DiGraph) -> dict:
    """
    Converts a networkx DiGraph object to a dictionary representation.

    **Parameters:**
    - G (nx.DiGraph): The input graph.

    **Returns:**
    - dict: A dictionary where each key is a node and the value is a list
            of its successor nodes (children).
    """
    return {node: list(G.successors(node)) for node in G.nodes()}


# --- Example Usage (can be added to the bottom of graph.py for testing) ---
if __name__ == '__main__':
    print("--- Testing the efficient generate_random_graph function ---")
    
    # Generate a graph
    graph_data = generate_random_graph(
        num_observables=15,
        num_unobservables=4,
        num_treatments=2,
        num_outcomes=2,
        sparcity_constant=0.15, # Equivalent to edge_prob=0.3
        seednum=101
    )
    
    graph_dict, positions, treatments, outcomes = graph_data
    
    print("\nGenerated Treatments:", treatments)
    print("Generated Outcomes:", outcomes)
    
    # Verify acyclicity by reconstructing the graph from the dictionary
    G_reconstructed = nx.DiGraph(graph_dict)

    is_acyclic = nx.is_directed_acyclic_graph(G_reconstructed)
    print(f"\nIs the full generated graph acyclic? {'Yes' if is_acyclic else 'No'}")
    
    print(f"Total nodes in graph: {len(G_reconstructed.nodes())}")
    print(f"Total edges in graph: {len(G_reconstructed.edges())}")