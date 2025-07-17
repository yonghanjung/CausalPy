import scipy.stats as stats
import networkx as nx
import numpy as np 
import random 
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

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

def Random_SCM_Generator2(num_observables, num_unobservables, num_treatments, num_outcomes, **kwargs):
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
		scm.generate_random_scm_test(num_observables, num_unobservables, num_treatments, num_outcomes, sparcity_constant, seednum, discrete )
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


def Random_SCM_Generator(num_observables, num_unobservables, num_treatments, num_outcomes, condition_ID = True, condition_BD = False, condition_mSBD = False, condition_FD = False, condition_Tian = False, condition_gTian = False, condition_product = False, discrete = True, seednum = 123):
	scm = SCM.StructuralCausalModel()

	store_graph = []
	stored_graph_hashes = set()
	ID_store_graph = []
	ID_store_graph_hashes = set()
	idx = 1 
	same_graph_idx = 1
	sparcity_constant = 0.75
	
	while 1:
		scm.generate_random_scm_test(num_observables, num_unobservables, num_treatments, num_outcomes, sparcity_constant, seednum, discrete )
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

def Random_Graph_Generator2(num_observables, num_unobservables, num_treatments, num_outcomes, **kwargs):
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

	np.random.seed(seednum)
	random.seed(seednum)

	store_graph = []
	stored_graph_hashes = set()
	ID_store_graph = []
	ID_store_graph_hashes = set()
	idx = 1 
	same_graph_idx = 1
	sparcity_constant = 0.75
	
	while 1: 
		[graph_dict, node_positions, X, Y] = graph.generate_random_graph(num_observables = num_observables, num_unobservables = num_unobservables, num_treatments = num_treatments, num_outcomes = num_outcomes, sparcity_constant = sparcity_constant)
		G = graph.create_acyclic_graph(graph_dict=graph_dict, an_Y_graph_TF = False, Y = None, node_positions = node_positions)
		# print(graph_dict, X, Y)
		
		if is_graph_in_list(G, stored_graph_hashes):
			# print("already checked", same_graph_idx)
			same_graph_idx += 1 
			if same_graph_idx > 500000:
				print("******* Cannot find a graph with given conditions *******")
				break 
			continue
		else:
			store_graph.append(G)
			stored_graph_hashes.add(get_graph_hash(G))
			G0, X0, Y0 = identify.preprocess_GXY_for_ID(G, X, Y)
			ID_result = identify.ID_return_Ctree(G, X, Y)

			if condition_ID is None: 
				return [graph_dict, node_positions, X, Y]

			if ID_result[0] == -1: 
				continue 
			elif ID_result[0] == 0: 
				if condition_ID == False: 
					return [graph_dict, node_positions, X, Y]
				else:
					continue 
			else:
				ID_store_graph.append(G)
				ID_store_graph.append(G)
				ID_store_graph_hashes.add(get_graph_hash(G))

				idx += 1 
				# print(idx, len(ID_store_graph))

				if conditions is False: 
					return [graph_dict, node_positions, X, Y]

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
					return [graph_dict, node_positions, X, Y]

def Random_Graph_Generator(num_observables, num_unobservables, num_treatments, num_outcomes, 
							condition_ID = True, condition_BD = False, condition_mSBD = False, condition_FD = False, condition_Tian = False, condition_gTian = False, condition_product = False,
							seednum = 123):
	''' Random graph generator '''
	np.random.seed(seednum)
	random.seed(seednum)

	store_graph = []
	stored_graph_hashes = set()
	ID_store_graph = []
	ID_store_graph_hashes = set()
	idx = 1 
	same_graph_idx = 1
	sparcity_constant = 0.75
	
	while 1: 
		[graph_dict, node_positions, X, Y] = graph.generate_random_graph(num_observables = num_observables, num_unobservables = num_unobservables, num_treatments = num_treatments, num_outcomes = num_outcomes, sparcity_constant = sparcity_constant)
		G = graph.create_acyclic_graph(graph_dict=graph_dict, an_Y_graph_TF = False, Y = None, node_positions = node_positions)
		# print(graph_dict, X, Y)
		if is_graph_in_list(G, stored_graph_hashes):
			print("already checked", same_graph_idx)
			same_graph_idx += 1 
			if same_graph_idx > 500000:
				print("******* Cannot find a graph with given conditions *******")
				return 
			continue
		else:
			store_graph.append(G)
			stored_graph_hashes.add(directional_dag_hash(G))
			G0, X0, Y0 = identify.preprocess_GXY_for_ID(G, X, Y)
			ID_result = identify.ID_return_Ctree(G, X, Y)

			if ID_result[0] == -1: 
				continue 
			elif ID_result[0] == 0: 
				if condition_ID == False: 
					return [graph_dict, node_positions, X, Y]
				else:
					continue 
			else:
				ID_store_graph.append(G)
				ID_store_graph.append(G)
				ID_store_graph_hashes.add(get_graph_hash(G))

				idx += 1 
				print(idx, len(ID_store_graph))

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
					return [graph_dict, node_positions, X, Y]