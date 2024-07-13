import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
import copy
import adjustment
import mSBD
import graph
import frontdoor
import tian
import plotly.graph_objects as go

def preprocess_GXY_for_ID(G, X, Y):
	''' 
	Graph Preprocessing for identification 
	Step 1. G = G(An_{G}(Y))
	Step 2. X = X intersection V
	step 3. X = X intersection An_{G_{bar{X}}(Y)}
	'''
	# Step 1. G = G(An_{G}(Y))
	G = graph.subgraphs(G, graph.find_ancestor(G,Y))
	
	# Step 2. X = X intersection V
	X = set(X).intersection(G.nodes)
	
	# step 3. X = X intersection An_{G_{bar{X}}(Y)}
	X = list( set(X).intersection(graph.find_ancestor(graph.G_cut_incoming_edges(G,X),Y)) )
	
	return [G, X, Y]

def ID_return_Ctree(G, X, Y):
	'''
	Identify P(Y | do(X)) from G 
	'''
	ID_status = -1

	G, X, Y = preprocess_GXY_for_ID(G, X, Y)
	if not X: 
		ID_status = [-1,-1,-1]
		return ID_status

	# Find c-components in G
	c_components = graph.list_all_c_components(G)

	# Find V\X and create the corresponding subgraph
	V_minus_X = list( set(G.nodes()).difference(set(X)) )
	subgraph_V_minus_X = graph.subgraphs(G,V_minus_X)

	# Find ancestors of Y in G(V\X)
	D = graph.find_ancestor(subgraph_V_minus_X,Y)

	# Find c-components in G(D)
	subgraph_D = graph.subgraphs(G,D)
	c_components_D = graph.list_all_c_components(subgraph_D)

	# Initialize dictionaries to store components and operations
	ID_dict_components = {i: [] for i in range(len(c_components_D))}
	ID_dict_operations = {i: [] for i in range(len(c_components_D))}

	unID_flag = False
	for i, Di in enumerate(c_components_D):
		Sj = graph.find_c_components(G, Di)
		c_components_series = [Sj]
		c_operations_series = ["\u03B4"] # delta 
		W = Sj
		# Repeat process to check identifiability
		while True:
			G_W = graph.subgraphs(G,W)
			A = graph.find_ancestor(G_W, Di)
			if set(A) == set(Di): 
				if Di != W:
					c_components_series.append(A)
					c_operations_series.append("\u03A3")
				break
			if set(A) == set(W):
				unID_flag = True
				break
			c_components_series.append(A)
			c_operations_series.append("\u03A3")
			subgraph_A = graph.subgraphs(G,A)
			S_ = graph.find_c_components(subgraph_A, Di)
			if S_ != A:
				c_components_series.append(S_)
				c_operations_series.append("\u03B4")
			W = S_

		if unID_flag:
			# print("P(Y | do(X)) is not identifiable from G because Q["f'{Di}'"] is not identifiable from G("f'{A}'")")
			return [0, Di, A]
			# raise ValueError("P(Y | do(X)) is not identifiable from G due to G("f'{A}'")")

		else:
			ID_dict_components[i] = c_components_series
			ID_dict_operations[i] = c_operations_series

	return [1, ID_dict_components, ID_dict_operations]

def draw_C_tree(G,X,Y,save_path = None):
	[ID_constant, ID_dict_components, ID_dict_operations] = ID_return_Ctree(G, X, Y)
	if ID_constant == -1: 
		# print("P(Y | do(X)) = P(Y)")
		return None 
	
	elif ID_constant != 1:
		return None 

	V = graph.find_topological_order(G)
	P_V = graph.write_joint_distribution(V)

	# Create an empty directed graph for the tree
	c_tree = nx.DiGraph()
	c_tree.add_node(P_V)  # Add P_V as the root node

	# Check if Q[D] needs to be made based on the number of components
	Q_D_made = False 
	if len(ID_dict_components) > 1: 
		Q_D_made = True 
		D = set()
		# Aggregate all last elements from the component lists in ID_dict_components
		for values in ID_dict_components.values():
			if values:
				last_element = values[-1]
				D.update(last_element)
		c_tree.add_node(f"Q{str(D)}")

	# Add the causal effect node
	c_tree.add_node(f"P({str(Y)} | do({str(X)}))")

	# Add edges based on components and operations
	for key, values in ID_dict_components.items():
		parent_node = P_V
		for idx in range(len(values)):
			value = values[idx]
			child_node = f"Q{str(value)}"
			c_tree.add_edge(parent_node, child_node, operation=ID_dict_operations[key][idx])
			parent_node = child_node
		
		# Connect to Q[D] or directly to the causal effect
		## \u03B4: \delta 
		## \u03A3: \Sigma 
		if Q_D_made:
			c_tree.add_edge(child_node, f"Q{str(D)}", operation="\u03B4")
		else: 
			operation = "=" if value == set(Y) else "\u03A3"
			c_tree.add_edge(child_node, f"P({str(Y)} | do({str(X)}))", operation=operation)

	# Connect Q[D] to the causal effect, if Q[D] is made
	if Q_D_made:
		operation = "=" if D == set(Y) else "\u03A3"
		c_tree.add_edge(f"Q{str(D)}", f"P({str(Y)} | do({str(X)}))", operation=operation)

	# Initialize the position dictionary
	## 
	pos = {}
	y_gap = 25.0  # Vertical gap between different keys
	x_gap = 50.0  # Horizontal gap between nodes within the same key
	y_pos = 0  # Initial y position

	y_pos = 0  # Initial y position
	for key, values in ID_dict_components.items():
		x_pos = 0  # Reset x position for each key
		for value in values:
			node_label = f"Q{str(value)}"
			pos[node_label] = (x_pos, y_pos)
			x_pos += x_gap  # Increment x position

		# Increment y position after processing each key
		y_pos += y_gap

	if Q_D_made:
		x_pos += 2*x_gap 
		pos[f"Q{str(D)}"] = (x_pos , 0)

	# Additional positions for P_V and causal effect node
	pos[P_V] = (-x_gap, 0)  # Position for P_V
	pos[f"P({str(Y)} | do({str(X)}))"] = (x_pos + x_gap, 0)  # Position for causal effect

	labels = nx.get_edge_attributes(c_tree, 'operation')

	edge_x = []
	edge_y = []
	for edge in c_tree.edges():
		x0, y0 = pos[edge[0]]
		x1, y1 = pos[edge[1]]
		edge_x.append(x0)
		edge_x.append(x1)
		edge_x.append(None)
		edge_y.append(y0)
		edge_y.append(y1)
		edge_y.append(None)
	
	edge_trace = go.Scatter(
		x=edge_x, y=edge_y,
		line=dict(width=0.5, color='#888'),
		hoverinfo='none',
		mode='lines')

	node_x = []
	node_y = []
	for node in c_tree.nodes():
		x, y = pos[node]
		node_x.append(x)
		node_y.append(y)
	
	node_trace = go.Scatter(
		x=node_x, y=node_y,
		mode='markers+text',
		text=list(c_tree.nodes()),
		textposition='top center',
		# hoverinfo='text',
		marker=dict(
			showscale=False,
			colorscale='YlGnBu',
			size=20,
			# colorbar=dict(
			# 	thickness=15,
			# 	title='Node Connections',
			# 	xanchor='left',
			# 	titleside='right'
			# ),
			line_width=2)
		)
	
	fig = go.Figure(data=[edge_trace, node_trace],
					layout=go.Layout(
						title='<br> C-tree',
						titlefont_size=30,
						showlegend=False,
						hovermode='closest',
						margin=dict(b=20, l=5, r=5, t=40),
						annotations=[dict(
							# text="C-tree",
							showarrow=True,
							xref="paper", yref="paper",
							x=0.005, y=-0.002 )],
						xaxis=dict(showgrid=False, zeroline=False, visible=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False, showticklabels=False)))


	fig.show()


# def new_identify_tree(X, Y, G):

# 	ID_status = -1

# 	G, X, Y = preprocess_GXY_for_ID(G, X, Y)
# 	if not X: 
# 		ID_status = [-1,-1,-1]
# 		return ID_status

# 	# condition_adjustment = adjustment.check_adjustment(G,X,Y)
# 	# condition_mSBD = mSBD.mSBD_criterion(G,X,Y)
# 	# condition_FD = frontdoor.find_frontdoor(G,X,Y)

# 	# if condition_adjustment or condition_mSBD or condition_FD:
# 	# 	return True 

# 	# Find V\X and create the corresponding subgraph
# 	c_components_D = prepare_c_components_D(X,Y,G)

# 	# Initialize dictionaries to store components and operations
# 	ID_dict_components = {i: [] for i in range(len(c_components_D))}
# 	ID_dict_operations = {i: [] for i in range(len(c_components_D))}

# 	# ID_flag = True
# 	for i, Di in enumerate(c_components_D):
# 		W = Di 
# 		curr_forest = graph.find_c_components(G,W).intersection(graph.find_ancestor(G,W))
# 		while True:
# 			prev_forest = copy.copy(curr_forest)
# 			graph_prev_forest = graph.subgraphs(G, prev_forest)
# 			curr_forest = graph.find_c_components(graph_prev_forest,W).intersection(graph.find_ancestor(graph_prev_forest,W))
# 			if prev_forest == curr_forest:
# 				break 

# 		if bool(set(X).intersection(curr_forest)):
# 			ID_status = [0,0,0]
# 			return ID_status # unID 

# 		pa_W = graph.find_parents(G, W)
# 		pa_W, W, ancestral_graph = prepare_XYG_for_ID(pa_W, W, G)
# 		S = graph.find_c_components(ancestral_graph, W)
# 		A = graph.find_ancestor( graph.subgraphs(ancestral_graph, S), W )
# 		while True: 
# 			ancestral_graph = graph.subgraphs(ancestral_graph, A)
# 			S = graph.find_c_components(ancestral_graph, W)
# 			A_dash = graph.find_ancestor( graph.subgraphs(ancestral_graph, S), W )
# 			if A == A_dash:
# 				break 
# 			pa_A_dash = graph.find_parents(G,A_dash)
# 			if not mSBD.mSBD_criterion(G, pa_A_dash, A_dash):
# 				break
# 			A = copy.copy(A_dash)

# 		# Identify Q[A] -> Q[W]
# 		ID_dict_components[i].append(A)
# 		if A == W: 
# 			continue 
		
# 		while True: 
# 			ancestral_graph = graph.subgraphs(ancestral_graph, A)
# 			S = graph.find_c_components(ancestral_graph, W)
# 			ID_dict_components[i].append(S)
# 			ID_dict_operations[i].append("\u03B4") # delta 
# 			A = graph.find_ancestor( graph.subgraphs(ancestral_graph, S), W )
# 			if A != S:
# 				ID_dict_components[i].append(A)
# 				ID_dict_operations[i].append("\u03A3") # Sigma 
# 			if A == W: 
# 				break 



# 		# adjustment_set_W = A
# 		# pa_adjustment_set_W = graph.find_parents(G,A)

					
# 	return [1, ID_dict_components, ID_dict_operations]



def draw_AC_tree(G, X, Y):
	[ID_constant, ID_dict_components, ID_dict_operations] = ID_return_Ctree(G, X, Y)
	if ID_constant == -1: 
		# print("P(Y | do(X)) = P(Y)")
		return None 
	
	elif ID_constant != 1:
		return None 

	# Initialize dictionaries to hold the adjusted components and operations
	adj_dict_components = {i: [] for i in ID_dict_components.keys()}
	adj_dict_operations = {i: [] for i in ID_dict_components.keys()}
	
	# Iterate over each component in ID_dict_components
	for key, values in ID_dict_components.items():
		# Reverse iteration to find the first element satisfying the mSBD criterion
		for elem in reversed(values):
			W = list(elem)  # Convert elem to list if it's not already
			R = list(graph.find_parents(G, W))  # Find the parents of Y in G
			# Check if the mSBD criterion is satisfied
			if mSBD.constructive_SAC_criterion(G, R, W):
				# If satisfied, clip the list at the current element
				index_to_keep = values.index(elem)
				adj_dict_components[key] = values[index_to_keep:]
				# Adjust the operations list accordingly
				# Operations start from the element before the current one
				adj_dict_operations[key] = ID_dict_operations[key][index_to_keep+1:]
				break
	
	# Create a new directed graph for the AC tree
	AC_tree = nx.DiGraph()

	# Initialize positions
	x_pos = 0
	y_pos = 0
	x_gap = 50
	y_gap = 25
	max_x = 0
	pos = {}  # Dictionary to store positions

	# Add nodes, edges, and set positions for each component
	for key, values in adj_dict_components.items():
		if not values:
			continue

		first_elem = f"A{values[0]}"
		AC_tree.add_node(first_elem)
		pos[first_elem] = (x_pos, y_pos)

		previous_node = first_elem
		for i, elem in enumerate(values[1:]):
			# Subsequent elements get the "Q" prefix
			current_node = f"Q{elem}"
			AC_tree.add_node(current_node)
			operation = adj_dict_operations[key][i]
			AC_tree.add_edge(previous_node, current_node, label=operation)

			# Update position
			x_pos += x_gap
			max_x = max(max_x, x_pos)
			pos[current_node] = (x_pos, y_pos)

			previous_node = current_node

		# Increment y_pos for the next key
		y_pos += y_gap
		x_pos = 0  # Reset x_pos for the next row

	# Add the special Q{D} node and position if needed
	Q_D_made = len(adj_dict_components) > 1
	if Q_D_made:
		D = set()
		for values in adj_dict_components.values():
			if values:
				last_element = values[-1]
				D.update(last_element)
		
		D_node = f"Q{str(D)}"
		AC_tree.add_node(D_node)
		x_pos = max_x + 1 * x_gap
		max_x = max(max_x, x_pos)
		pos[D_node] = (x_pos, 0)

		for key, values in adj_dict_components.items():
			if values:
				# Check if the last element is the first in its list
				last_elem_prefix = "A" if len(values) == 1 else "Q"
				last_elem_node = f"{last_elem_prefix}{values[-1]}"
				AC_tree.add_edge(last_elem_node, D_node, label="\u03B4")

		operation = "=" if D == set(Y) else "\u03A3"
		AC_tree.add_edge(D_node, f"P({str(Y)} | do({str(X)}))", label=operation)
	else:
		# Check if the last element is the first in its list
		last_elem_prefix = "A" if len(adj_dict_components[0]) == 1 else "Q"
		last_elem_node = f"{last_elem_prefix}{adj_dict_components[0][-1]}"
		operation = "=" if adj_dict_components[list(adj_dict_components.keys())[0]][-1] == set(Y) else "\u03B4"
		AC_tree.add_edge(last_elem_node, f"P({str(Y)} | do({str(X)}))", label=operation)

	# Position for causal effect node
	pos[f"P({str(Y)} | do({str(X)}))"] = (max_x + 1 * x_gap, 0)
	labels = nx.get_edge_attributes(AC_tree, 'label')

	edge_x = []
	edge_y = []
	for edge in AC_tree.edges():
		x0, y0 = pos[edge[0]]
		x1, y1 = pos[edge[1]]
		edge_x.append(x0)
		edge_x.append(x1)
		edge_x.append(None)
		edge_y.append(y0)
		edge_y.append(y1)
		edge_y.append(None)
	
	edge_trace = go.Scatter(
		x=edge_x, y=edge_y,
		line=dict(width=0.5, color='#888'),
		hoverinfo='none',
		mode='lines')

	node_x = []
	node_y = []
	for node in AC_tree.nodes():
		x, y = pos[node]
		node_x.append(x)
		node_y.append(y)
	
	node_trace = go.Scatter(
		x=node_x, y=node_y,
		mode='markers+text',
		text=list(AC_tree.nodes()),
		textposition='top center',
		# hoverinfo='text',
		marker=dict(
			showscale=False,
			colorscale='YlGnBu',
			size=20,
			# colorbar=dict(
			# 	thickness=15,
			# 	title='Node Connections',
			# 	xanchor='left',
			# 	titleside='right'
			# ),
			line_width=2)
		)
	
	fig = go.Figure(data=[edge_trace, node_trace],
					layout=go.Layout(
						title='<br> AC-tree',
						titlefont_size=30,
						showlegend=False,
						hovermode='closest',
						margin=dict(b=20, l=5, r=5, t=40),
						annotations=[dict(
							# text="C-tree",
							showarrow=True,
							xref="paper", yref="paper",
							x=0.005, y=-0.002 )],
						xaxis=dict(showgrid=False, zeroline=False, visible=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False, showticklabels=False)))


	fig.show()



def causal_identification(G,X,Y,latex = True):
	def compute_Q_W(G, W, topo_W, latex):
		R = list(graph.find_parents(G, topo_W))
		if len(R) == 0:
			Q_W = graph.write_joint_distribution( topo_W )
		else:
			Q_W = mSBD.mSBD_estimand(G, R, topo_W, latex, True)
		return Q_W

	def print_estimand(causal_expression, latex):
		if latex:
			plt.figure("estimand",figsize=(15, 6))
			plt.text(0.5, 0.5, f'${causal_expression}$', fontsize=15, ha='center')
			plt.axis('off')
			plt.show(block=False)
			return None
		else: 
			return causal_expression
	
	# Prepare "X, Y" to be printed 
	topo_V = graph.find_topological_order(G)
	X = sorted(X, key = lambda x: topo_V.index(x))
	Y = sorted(Y, key = lambda x: topo_V.index(x))

	X_print = ', '.join(X)
	Y_print = ', '.join(Y)
	orig_X = X 

	# Run the ID algorithm 
	ID_constant, ID_dict_components, ID_dict_operations = ID_return_Ctree(G,X,Y)

	
	# Return if P(Y | do(x)) = P(Y)
	if ID_constant == -1:
		print(f"P({Y_print} | do({X_print})) = P({Y_print})")
		return None

	# Return if P(Y | do(x)) is not ID 
	elif ID_constant == 0:
		print( f"P({Y_print} | do({X_print})) is not identifiable from G, since Q[{ID_dict_components}] is not identifiable from G({ID_dict_operations})")
		return None

	# Preprocess the graph 
	G, X, Y = preprocess_GXY_for_ID(G, X, Y)

	# simplified X
	X_simplified_print = ', '.join(X)

	# causal_expression
	causal_expression = f"P({Y_print} | do({X_print}))"
	if orig_X != X and len(orig_X) != len(X):
		causal_expression += f" = P({Y_print} | do({X_simplified_print}))"

	# Check if P(Y | do(x)) can be estimated through the adjustment
	condition_adjustment = adjustment.check_admissibility(G,X,Y)
	if condition_adjustment:
		Z = adjustment.construct_minimum_adjustment_set(G, X, Y)
		message = f"{Z} is admisslbe w.r.t. {X} and {Y} in G"
		estimand = f" = {adjustment.adjustment_estimand(X,Y,Z, latex)}"
		causal_expression += estimand
		print(message)
		return print_estimand(causal_expression, latex)

	# Check if P(Y | do(x)) can be estimated through the mSBD
	condition_mSBD = mSBD.constructive_SAC_criterion(G, X, Y)
	if condition_mSBD:
		dict_X, dict_Z, dict_Y = mSBD.check_SAC_with_results(G,X,Y, minimum = True)
		message = f"{{{dict_Z}}} is mSBD admisslbe w.r.t. {dict_X} and {dict_Y} in G"
		estimand = f" = {mSBD.mSBD_estimand(G, X, Y, latex, minimum=True)}"
		causal_expression += estimand
		print(message)
		return print_estimand(causal_expression, latex)
		
	# Check if P(Y | do(x)) can be estimated through the FD
	condition_FD = frontdoor.constructive_minimum_FD(G,X,Y)
	if condition_FD:
		ZC = condition_FD.copy()
		Z = ZC['Z']
		C = ZC['C']
		message = f"{{{ZC}}} is FD admisslbe w.r.t. {X} and {Y} in G"
		estimand = f" = {frontdoor.frontdoor_estimand(X,Y,Z,C,latex)}"
		causal_expression += estimand
		print(message)
		return print_estimand(causal_expression, latex)

	# Check if P(Y | do(x)) can be estimated through the Tian's adjustment
	condition_Tian = tian.check_Tian_criterion(G, X)
	if condition_Tian:
		message = f"Tian's adjustment is satisfied w.r.t. {X} and {Y} in G"
		estimand = f" = {tian.Tian_estimand(G, X, Y, latex, topo_V = None)}"
		causal_expression += estimand
		print(message)
		return print_estimand(causal_expression, latex)

	# Check if P(Y | do(x)) can be estimated through the Generalized Tian's adjustment
	condition_generalized_Tian = tian.check_Generalized_Tian_criterion(G, X)
	if condition_generalized_Tian:
		message = f"Generalized Tian's adjustment is satisfied w.r.t. {X} and {Y} in G"
		estimand = f" = {tian.generalized_Tian_estimand(G, X, Y, latex, topo_V = None)}"
		causal_expression += estimand
		print(message)
		return print_estimand(causal_expression, latex)

	
	# Initialize dictionaries to hold the adjusted components and operations
	adj_dict_components = {i: [] for i in ID_dict_components.keys()}
	adj_dict_operations = {i: [] for i in ID_dict_components.keys()}
	
	# Iterate over each component in ID_dict_components
	for key, values in ID_dict_components.items():
		# Reverse iteration to find the first element satisfying the mSBD criterion
		for elem in reversed(values):
			W = sorted(list(elem), key = lambda x: topo_V.index(x)) # Convert elem to list if it's not already
			# W = list(elem)  
			R = list(graph.find_parents(G, W))  # Find the parents of Y in G
			# Check if the mSBD criterion is satisfied
			if mSBD.constructive_SAC_criterion(G, R, W):
				# If satisfied, clip the list at the current element
				index_to_keep = values.index(elem)
				adj_dict_components[key] = values[index_to_keep:]
				# Adjust the operations list accordingly
				# Operations start from the element before the current one
				adj_dict_operations[key] = ID_dict_operations[key][index_to_keep+1:]
				break

	dict_mSBD_TF = {i: [] for i in adj_dict_components.keys()}
	Q_Di = {i: [] for i in adj_dict_components.keys()}
	for key, values in adj_dict_components.items():
		W = list(values[0])
		topo_W = list(graph.find_topological_order( graph.subgraphs(G, W) ))
		Q_W = compute_Q_W(G, W, topo_W, latex)
		Q_Di[key].append(Q_W)
		dict_mSBD_TF[key].append(True)
		
		for i, elem in enumerate(values[1:]):
			idx = i+1
			C = list(values[idx])
			C_values = ', '.join(char.lower() for char in C)
			operation = adj_dict_operations[key][idx-1]
			Q_W_mSBD_True_False = dict_mSBD_TF[key][idx-1]

			# Q_C = compute_Q_C(G, W, C, topo_W, Q_W, operation, Q_W_mSBD_True_False, latex)
			if operation == "\u03A3": # \sum
				if Q_W_mSBD_True_False: # Q[C] = \sum Q[W] where Q[W] is mSBD
					dict_mSBD_TF[key].append(True) # Q[C] = \sum Q[W] where Q[W] is mSBD
					R = list(graph.find_parents(G, C))
					Q_C = mSBD.mSBD_estimand(G, R, C, latex, minimum = True)
					
					W = C.copy()
					topo_W = list(graph.find_topological_order( graph.subgraphs(G, W) ))
					Q_W = Q_C[:]
					Q_Di[key].append(Q_C)
					continue
				else: # Q[C] = \sum Q[W], where Q[W] is non-mSBD
					dict_mSBD_TF[key].append(False) # Q[C] is not mSBD
					W_minus_C = list(set(W).difference(set(C)))
					W_minus_C_lower = [char.lower() for char in W_minus_C]
					W_minus_C_values = ', '.join(W_minus_C_lower)
					if not latex:
						Q_C = f"\u03A3_{{{W_minus_C_values}}}{Q_W}"
					else: 
						Q_C = f"\\sum_{{{W_minus_C_values}}} {Q_W}"
					W = C.copy()
					topo_W = list(graph.find_topological_order( graph.subgraphs(G, W) ))
					Q_W = Q_C[:]
					Q_Di[key].append(Q_C)
					continue
			elif operation == "\u03B4":  # \delta 
				dict_mSBD_TF[key].append(False) #Q[C] is not mSBD admissible 
				if Q_W_mSBD_True_False: # Q[W] is mSBD.  Q[C] = Q[W]/Q[W\C] where 
					W_minus_C = list(set(W).difference(set(C)))
					R = list(graph.find_parents(G, W_minus_C))
					if mSBD.constructive_SAC_criterion(G, R, W_minus_C): # if Q[W\C] is mSBD
						Q_W_C = mSBD.mSBD_estimand(G, R, W_minus_C, latex, minimum = True)
						if not latex:
							Q_C = f"[[{Q_W}]/[{Q_W_C}]]"
						else:
							Q_C = f"\\frac{{{Q_W}}}{{{Q_W_C}}}"
						W = C.copy()
						topo_W = list(graph.find_topological_order( graph.subgraphs(G, W) ))
						Q_W = Q_C[:]
						Q_Di[key].append(Q_C)
						continue
					else: # if Q[W\C] is NOT mSBD but Q[W] is mSBD
						Q_C_component = []
						for c in C: 
							c_index = topo_W.index(c)       
							Wi_to_Wm = topo_W[c_index:]
							Wi1_to_Wm = topo_W[c_index+1:]
							# Q[W1,...,W{i}]
							W_1_to_i = list(set(W).difference(set(Wi1_to_Wm)))
							R_W_1_to_i = list(graph.find_parents(G, W_1_to_i))
							numerator = mSBD.mSBD_estimand(G, R_W_1_to_i, W_1_to_i, latex, minimum = True)
							# Q[W1,...,W{i-1}]
							W_1_to_i1 = list(set(W).difference(set(Wi_to_Wm))) 
							if len(W_1_to_i1) == 0:
								Q_C_component_element = f"{numerator}"
								Q_C_component.append(Q_C_component_element)
							else:
								R_W_1_to_i1 = list(graph.find_parents(G, W_1_to_i1))
								denominator = mSBD.mSBD_estimand(G, R_W_1_to_i1, W_1_to_i1, latex, minimum = True)
								if not latex: 
									Q_C_component_element = f"[({numerator}) / ({denominator})]"
								else:
									Q_C_component_element = f"\\frac{{{numerator}}}{{{denominator}}}"
								Q_C_component.append(Q_C_component_element)
						if len(C) > 1:
							if not latex:
								Q_C = "*".join(reversed(Q_C_component))
							else:
								Q_C = " ".join(reversed(Q_C_component))
						else:
							Q_C = Q_C_component[0]
						Q_Di[key].append(Q_C)
						W = C.copy()
						topo_W = list(graph.find_topological_order( graph.subgraphs(G, W) ))
						Q_W = Q_C[:]
						continue
				else: # Q[C] = Q[W]/Q[W\C] where Q[W] is NOT mSBD
					Q_C_component = []
					for c in C: 
						c_index = topo_W.index(c)       
						Wi_to_Wm = topo_W[c_index:]
						Wi1_to_Wm = topo_W[c_index+1:]
						Wi_to_Wm_lower = [char.lower() for char in Wi_to_Wm]
						Wi_to_Wm_summands = ', '.join(Wi_to_Wm_lower)
						Wi1_to_Wm_lower = [char.lower() for char in Wi1_to_Wm]
						Wi1_to_Wm_summands = ', '.join(Wi1_to_Wm_lower)
						if len(Wi1_to_Wm) == 0:
							numerator = Q_W 
						else: 
							if not latex:
								numerator = f"\u03A3_{{{Wi1_to_Wm_summands}}}{Q_W}"
							else:
								numerator = f"\\sum_{{{Wi1_to_Wm_summands}}} {Q_W}"
						if len(Wi_to_Wm) == 0:
							denominator = Q_W
						else: 
							if not latex:
								denominator = f"\u03A3_{{{Wi_to_Wm_summands}}}{Q_W}"
							else:
								denominator = f"\\sum_{{{Wi_to_Wm_summands}}} {Q_W}"
						if not latex: 
							Q_C_element = f"[({numerator}) / ({denominator})]"
						else: 
							Q_C_element = f"\\frac{{{numerator}}}{{{denominator}}}"
						Q_C_component.append(Q_C_element)
					if len(C) > 1:
						if not latex:
							Q_C = "*".join(reversed(Q_C_component))
						else:
							Q_C = " ".join(reversed(Q_C_component))
					else:
						Q_C = Q_C_component[0]
					Q_Di[key].append(Q_C)
					W = C.copy()
					topo_W = list(graph.find_topological_order( graph.subgraphs(G, W) ))
					Q_W = Q_C[:]
					continue
	if len(Q_Di) > 1: 
		Q_Di_list = []
		for key, values in Q_Di.items():
			if not latex:
				last_elem = "[" + Q_Di[key][-1] + "]"
			else:
				last_elem = f"\\left({{{Q_Di[key][-1]}}}\\right)"
			Q_Di_list.append(last_elem)
		if not latex:
			Q_D = "*".join(Q_Di_list)
		else:
			Q_D = " ".join(Q_Di_list)
	else:
		Q_D = Q_Di[0][-1]

	# Find V\X and create the corresponding subgraph
	V_minus_X = list(set(G.nodes()).difference(set(X)))
	subgraph_V_minus_X = graph.subgraphs(G,V_minus_X)
	# Find ancestors of Y in G(V\X)
	D = list(set(graph.find_ancestor(subgraph_V_minus_X,Y)) | set(Y))

	if set(D) == set(Y):
		# print("hi")
		if not latex: 
			if orig_X != list(X) and len(orig_X) != len(X):
				causal_expression = f"P({{{Y_print}}} | do({{{X_print}}})) = P({{{Y_print}}} | do({{{X_simplified_print}}})) = {Q_D}"
			else:
				causal_expression = f"P({{{Y_print}}} | do({{{X_print}}})) = {Q_D}"
		else:
			if orig_X != list(X) and len(orig_X) != len(X):
				causal_expression = f"P({{{Y_print}}} \\mid do({{{X_print}}})) = P({{{Y_print}}} \\mid do({{{X_simplified_print}}})) = {Q_D}"
			else:
				causal_expression = f"P({{{Y_print}}} \\mid do({{{X_print}}})) = {Q_D}"
	else:
		# print("ho")
		D_minus_Y = list(set(D).difference(set(Y)))
		D_minus_Y_lower = [char.lower() for char in D_minus_Y]
		D_minus_Y_values = ', '.join(D_minus_Y_lower)
		if not latex:
			if orig_X != list(X) and len(orig_X) != len(X):
				causal_expression = f"P({{{Y_print}}} | do({{{X_print}}})) = P({{{Y_print}}} | do({{{X_simplified_print}}})) = \u03A3_{{{D_minus_Y_values}}}{Q_D}"
			else:
				causal_expression = f"P({{{Y_print}}} | do({{{X_print}}})) = \u03A3_{{{D_minus_Y_values}}}{Q_D}"
		else:
			if orig_X != list(X) and len(orig_X) != len(X):
				causal_expression = f"P({{{Y_print}}} \\mid do({{{X_print}}})) = P({{{Y_print}}} \\mid do({{{X_simplified_print}}})) = \\sum_{{{D_minus_Y_values}}} {Q_D}"
			else:
				causal_expression = f"P({{{Y_print}}} \\mid do({{{X_print}}})) = \\sum_{{{D_minus_Y_values}}} {Q_D}"

	return print_estimand(causal_expression, latex)




	





