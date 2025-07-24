import scipy.stats as stats
import itertools
import networkx as nx
import numpy as np 
import random 
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
import pyperclip

import graph
import SCM
import adjustment
import identify
import mSBD
import frontdoor
import tian

def get_graph_hash(graph):
	return weisfeiler_lehman_graph_hash(graph)

def directional_dag_hash(G: nx.DiGraph) -> str:
	"""Compute a Weisfeiler-Lehman hash that respects edge direction."""
	G_copy = nx.DiGraph()
	for u, v in G.edges():
		G_copy.add_edge(u, v, direction='forward')  # label direction
	return weisfeiler_lehman_graph_hash(G_copy, edge_attr='direction')

def is_graph_in_list(new_graph, stored_graph_hashes):
	new_graph_hash = directional_dag_hash(new_graph)
	if new_graph_hash in stored_graph_hashes:
		return True
	return False

def random_SCM_generator(num_observables, num_unobservables, num_treatments, num_outcomes, **kwargs):
	scm = SCM.StructuralCausalModel()

	''' Random graph generator '''
	seednum = kwargs.get('seednum', 123)
	condition_ID = kwargs.get('condition_ID', None)
	
	conditions = kwargs.get('conditions', False)

	condition_BD = kwargs.get('condition_BD', None)
	condition_mSBD = kwargs.get('condition_mSBD', None)
	condition_FD = kwargs.get('condition_FD', None)
	condition_Tian = kwargs.get('condition_Tian', None)
	condition_gTian = kwargs.get('condition_gTian', None)
	condition_product = kwargs.get('condition_product', None)
	discrete = kwargs.get('condition_product', True)

	store_graph = []
	stored_graph_hashes = set()
	ID_store_graph = []
	ID_store_graph_hashes = set()
	idx = 1 
	same_graph_idx = 1
	sparcity_constant = 0.75
	
	while 1:
		scm.generate_random_scm(num_observables, num_unobservables, num_treatments, num_outcomes, sparcity_constant, seednum, discrete )
		G = scm.graph
		X = [v for v in G.nodes if v.startswith('X')]
		Y = [v for v in G.nodes if v.startswith('Y')]

		if is_graph_in_list(G, stored_graph_hashes):
			# print("already checked", same_graph_idx)
			same_graph_idx += 1 
			if same_graph_idx > 50000:
				print("******* Cannot find a graph with given conditions *******")
				break 
			continue
		else:
			store_graph.append(G)
			stored_graph_hashes.add(get_graph_hash(G))
			G0, X0, Y0 = identify.preprocess_GXY_for_ID(G, X, Y)
			ID_result = identify.ID_return_Ctree(G, X, Y)

			if condition_ID is None: 
				return [scm, X, Y]

			if ID_result[0] == -1: 
				continue 
			elif ID_result[0] == 0: 
				if condition_ID == False: 
					return [scm, X, Y]
				else:
					continue 
			else:
				ID_store_graph.append(G)
				ID_store_graph.append(G)
				ID_store_graph_hashes.add(get_graph_hash(G))

				idx += 1 
				# print(idx, len(ID_store_graph))
				if conditions is False: 
					return [scm, X, Y]

				satisfied_adjustment = adjustment.check_admissibility(G0, X0, Y0)
				satisfied_mSBD = mSBD.constructive_SAC_criterion(G0, X0, Y0)
				satisfied_FD = frontdoor.constructive_FD(G0, X0, Y0)
				satisfied_Tian = tian.check_Tian_criterion(G0,X0)
				satisfied_gTian = tian.check_Generalized_Tian_criterion(G0,X0)
				satisfied_product = tian.check_product_criterion(G0, X0, Y0)

				# Create a dictionary to map conditions to their satisfaction checks
				condition_checks = {
					"condition_BD": satisfied_adjustment,
					"condition_mSBD": satisfied_mSBD,
					"condition_FD": satisfied_FD,
					"condition_Tian": satisfied_Tian,
					"condition_gTian": satisfied_gTian,
					"condition_product": satisfied_product
				}

				# Check if all specified conditions are satisfied
				all_conditions_met = True
				for condition, satisfied in condition_checks.items():
					condition_value = eval(condition)
					if condition_value and not satisfied:
						all_conditions_met = False
						break
					if not condition_value and satisfied:
						all_conditions_met = False
						break

				if all_conditions_met:
					return [scm, X, Y]

def get_adjacency_string(G: nx.DiGraph) -> bytes:
	"""
	Creates a unique, hashable byte string representing the graph's
	adjacency matrix with nodes in a fixed sorted order.
	"""
	# Get a sorted list of all node names to ensure a canonical order
	sorted_nodes = sorted(list(G.nodes()))
	# Generate the numpy adjacency matrix in that specific order
	adj_matrix = nx.to_numpy_array(G, nodelist=sorted_nodes)
	# Convert the numpy array to a compact byte string to be stored in a set
	return adj_matrix.tobytes()

def random_graph_generator(num_observables, num_unobservables, num_treatments, num_outcomes, **kwargs):
	''' Random graph generator '''
	# The main seed makes the sequence of generated graphs reproducible.
	main_seed = kwargs.get('seednum', 123)
	master_rng = random.Random(main_seed)
 
	# Bound the num_unobservables 
	max_unobservables = num_observables * (num_observables - 1) // 2
	if num_unobservables > max_unobservables:
		num_unobservables = max_unobservables
	

	# --- Parameters for search control ---
	max_graphs_to_test = kwargs.get('max_graphs', 1e7)
	max_consecutive_duplicates = kwargs.get('max_consecutive_duplicates', 10000)
	max_retries = kwargs.get('max_retries', 50)

	# --- Get user-defined conditions from kwargs ---
	condition_ID = kwargs.get('condition_ID', None)
	condition_BD = kwargs.get('condition_BD', None)
	condition_mSBD = kwargs.get('condition_mSBD', None)
	condition_FD = kwargs.get('condition_FD', None)
	condition_Tian = kwargs.get('condition_Tian', None)
	condition_gTian = kwargs.get('condition_gTian', None)
	condition_product = kwargs.get('condition_product', None)

	# --- Outer Retry Loop: Resets the main seed on failure ---
	for retry_attempt in range(max_retries):
		# Use the initial seed on the first try, then a random one for retries.
		current_seed = master_rng.randint(0, 1e7)
		random.seed(current_seed)
		np.random.seed(current_seed)
  
		print(f"\n--- Starting Search Attempt {retry_attempt + 1}/{max_retries} (Seed: {current_seed}) ---")

		# --- Initialize counters and storage ---
		stored_adj_strings = set()
		graphs_tested = 0
		consecutive_duplicates = 0

		while True:
			# --- Check exit conditions first ---
			if graphs_tested >= max_graphs_to_test:
				print(f"\nSearch stopped: Limit of {max_graphs_to_test} unique graphs tested was reached.")
				break
			if max_consecutive_duplicates and consecutive_duplicates > max_consecutive_duplicates:
				print(f"\nSearch stopped: Failed to find a new unique graph after {consecutive_duplicates} consecutive attempts.")
				break

			# --- Generate a new graph ---
			# Each graph gets its own seed from the main random generator's sequence.
			graph_seed = random.randint(0, 1e7)
   
			if kwargs.get('sparcity_constant') is None:
				np.random.seed(graph_seed)
				sparcity_constant = np.random.uniform(0.0, 1.0)
	
			[graph_dict, node_positions, X, Y] = graph.generate_random_graph(
				num_observables=num_observables,
				num_unobservables=num_unobservables,
				num_treatments=num_treatments,
				num_outcomes=num_outcomes,
				sparcity_constant=sparcity_constant,
				seednum=graph_seed
			)
			G = graph.create_acyclic_graph(graph_dict=graph_dict, node_positions=node_positions)
			adj_string = get_adjacency_string(G)
	
			# --- Check for duplicates ---
			if adj_string in stored_adj_strings:
				consecutive_duplicates += 1
				continue
			
			# --- Process the new, unique graph ---
			consecutive_duplicates = 0 # Reset counter
			stored_adj_strings.add(adj_string)
			graphs_tested += 1
	
			# --- Provide better progress feedback ---
			print(f"Unique graphs tested: {graphs_tested} | Consecutive duplicates: {consecutive_duplicates}   ", end='\r')

			# --- Filtering Logic (your original logic was correct here) ---
			id_status = identify.ID_return_Ctree(G, X, Y)[0] # 0: unID, 1: ID, -1: trivialID 
	
			if condition_ID is None:
				print(f"\nFound graph after testing {graphs_tested} unique graphs.")
				return [graph_dict, node_positions, X, Y]

			if id_status == -1:
				continue
			elif id_status == 0:
				if condition_ID is not None and condition_ID is False:
					print(f"\nFound non-identifiable graph after testing {graphs_tested} unique graphs.")
					return [graph_dict, node_positions, X, Y]
				else:
					continue
			else: # id_status is 1 (identifiable)
				G0, X0, Y0 = identify.preprocess_GXY_for_ID(G, X, Y)
				satisfied_adjustment = adjustment.check_admissibility(G0, X0, Y0)
				satisfied_mSBD = mSBD.constructive_SAC_criterion(G0, X0, Y0)
				satisfied_FD = frontdoor.check_FD(G0, X0, Y0)
				satisfied_Tian = tian.check_Tian_criterion(G0, X0)
				satisfied_gTian = tian.check_Generalized_Tian_criterion(G0, X0)
				satisfied_product = tian.check_product_criterion(G0, X0, Y0)

				condition_checks = {
					"condition_BD": satisfied_adjustment, "condition_mSBD": satisfied_mSBD,
					"condition_FD": satisfied_FD, "condition_Tian": satisfied_Tian,
					"condition_gTian": satisfied_gTian, "condition_product": satisfied_product
				}

				all_conditions_met = True
				for condition_name, graph_satisfies_criterion in condition_checks.items():	
					# First, check if the user actually specified a requirement for this criterion.
					user_request = kwargs.get(condition_name, None)

					# If the user did not pass this argument (e.g., 'condition_FD' was not in the
					# function call), then user_request is None. We don't need to check it,
					# so we skip to the next criterion in the loop.
					if user_request is None:
						continue

					# If we are here, it means the user has a specific request (e.g., condition_FD=True).
					# Now, we check for a mismatch between the user's request and the graph's property.

					# Case 1: The user wanted the condition to be TRUE, but it was FALSE.
					# This is a failure.
					if user_request is True and not graph_satisfies_criterion:
						all_conditions_met = False
						# Since one condition has failed, we can stop checking the rest.
						break

					# Case 2: The user wanted the condition to be FALSE, but it was TRUE.
					# This is also a failure.
					elif user_request is False and graph_satisfies_criterion:
						all_conditions_met = False
						# Since one condition has failed, we can stop checking the rest.
						break

					# If neither of the above failure cases were triggered, it means the graph
					# matches the user's request for this specific criterion, so we continue
					# the loop to check the next one.
				
				if all_conditions_met:
					print(f"\nFound graph with all conditions met after testing {graphs_tested} unique graphs.")
					return [graph_dict, node_positions, X, Y]
	
 # This part is reached only if all retry attempts have failed
	print("\nAll search attempts failed to find a matching graph.")
	return None

def find_graph_by_search(max_observables, max_unobservables, num_treatments, num_outcomes, **kwargs):
	"""
	Searches for a graph that satisfies the given criteria by iterating through
	different numbers of observable and unobservable nodes.

	This function acts as a manager, calling the `random_graph_generator` worker
	with different parameters until a suitable graph is found.

	Parameters:
	- max_observables (int): The maximum number of total observable nodes (V, X, Y).
	- max_unobservables (int): The maximum number of unobserved confounders.
	- num_treatments (int): The fixed number of treatment variables (X).
	- num_outcomes (int): The fixed number of outcome variables (Y).
	- **kwargs: All other condition flags (e.g., condition_ID=True) to be passed
				down to the worker function.
	"""
	min_observables = kwargs.get('min_observables',num_treatments + num_outcomes)
	if max_observables < min_observables:
		print(f"Error: max_observables ({max_observables}) cannot be less than the min_observables ({min_observables}).")
		return None

	min_unobservables = kwargs.get('min_unobservables',0)
	if max_unobservables < min_unobservables:
		print(f"Error: max_observables ({max_unobservables}) cannot be less than the min_unobservables ({min_unobservables}).")
		return None

	print("--- Starting Graph Search ---")
	
	# Iterate through the number of observable nodes, from simplest to most complex
	for n_obs in range(min_observables, max_observables + 1):
		# --- IMPROVED LOGIC ---
		# Calculate the theoretical maximum number of unobservables for n_obs nodes (n_obs choose 2).
		max_possible_unobs_for_n_obs = n_obs * (n_obs - 1) // 2
		
		
		# The actual upper bound for the inner loop is the smaller of the user-defined max
		# and the theoretical max for the current number of observables.
		actual_max_unobs = min(max_unobservables, max_possible_unobs_for_n_obs)

		# Iterate through the number of unobservable nodes up to the calculated limit
		for n_unobs in range(min_unobservables, actual_max_unobs + 1):
			print(f"\nSearching with Parameters: N_obs={n_obs}, N_unobs={n_unobs} (Max possible for N_obs={n_obs} is {max_possible_unobs_for_n_obs})")
			
			# Call the worker function with the current parameters
			result = random_graph_generator(
				num_observables=n_obs,
				num_unobservables=n_unobs,
				num_treatments=num_treatments,
				num_outcomes=num_outcomes,
				**kwargs
			)
			
			# If the worker found a graph that matches all criteria, we're done!
			if result is not None:
				print("\n--- Search Successful! ---")
				print(f"Found matching graph with N_obs={n_obs}, N_unobs={n_unobs}")
				return result

	# If the loops complete without finding a graph, the search has failed
	print("\n--- Search Failed ---")
	print("Could not find a graph matching the criteria within the specified parameter ranges.")
	return None


if __name__ == "__main__":
	seednum = 190602
	np.random.seed(seednum)
	random.seed(seednum)
 
	result = find_graph_by_search(
		min_observables=4,      # Min total observables (V+X+Y)
		max_observables=6,      # Max total observables (V+X+Y)
		min_unobservables=0,		# Min total unobservables 
		max_unobservables=3,    # Max unobservables
		num_treatments=1,       # Fixed number of treatments
		num_outcomes=1,         # Fixed number of outcomes
		condition_ID=True,
		# condition_BD=False,
		# condition_mSBD=False,
		# condition_FD=False,
		condition_Tian=False,
		# condition_gTian=False,
		condition_product=True,
		seednum=seednum
	)
 
	# result = random_graph_generator(
	#  			num_observables = 4, 
	#     		num_unobservables = 2, 
	#       		num_treatments = 1, 
	#         	num_outcomes = 1, 
	# 			# condition_ID = True, 
	# 			# condition_BD = True, 
	# 			# condition_mSBD = True, 
	# 			condition_FD = True, 
	# 			# condition_Tian = False, 
	# 			# condition_gTian = False, 
	# 			# condition_product = False, 
	# 			seednum = seednum)
	
 	#Check if the search was successful before unpacking
	if result is None:
	 	# Handle the failure case
		print("Search failed to find a matching graph.")
	else:
		graph_dict, node_positions, X, Y = result
		# Now you can proceed with the graph...
		print("Successfully found a graph!")

		G = graph.create_acyclic_graph(graph_dict=graph_dict, an_Y_graph_TF = False, Y = None, node_positions = node_positions)

		# Generate the random SCM 
		# [scm, X, Y] = random_generator.Random_SCM_Generator(num_observables = 5, num_unobservables = 3, num_treatments = 2, num_outcomes = 1, 
		# 																		condition_ID = True, condition_BD = False, condition_mSBD = False, condition_FD = False, condition_Tian = False, condition_gTian = True)
		# sample_data = scm.generate_samples(10000)[topo_V]
		# print(sample_data)
		# G = scm.graph
		

		# Visualize the graph 
		# graph.visualize(G)
		
		# Identify the causal effect P(Y | do(X)) from G 
		# G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
		print( identify.causal_identification(G,X,Y, latex = False, copyTF=True) )


		# Draw the C-tree and AC-tree 
		# identify.draw_C_tree(G,X,Y)
		# identify.draw_AC_tree(G,X,Y)

		adj_dict_components, adj_dict_operations = identify.return_AC_tree(G, X, Y)

		# Copy the graph for comparing with Fusion
		pyperclip.copy(graph.graph_dict_to_fusion_graph(graph_dict))