import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import KFold
import xgboost as xgb
import copy
from scipy.stats import spearmanr
from scipy.stats import norm
import warnings
import random
import time

import itertools
import random_generator
import graph
import identify
import adjustment
import frontdoor
import mSBD
import tian
import statmodules

import identify
import est_mSBD
import example_SCM
	
# Turn off alarms
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")
warnings.simplefilter(action='ignore', category=FutureWarning)

def add_cluster_to_vec(cluster_variables, vec):
	vec_new = []
	if cluster_variables is not None:
		for node in vec:
			if node in cluster_variables:
				# Expand cluster variable: find columns starting with the node, excluding the node itself
				expanded = [col for col in obs_data.columns if col.startswith(node) and col != node]
				# Sort the expanded columns to ensure proper ordering (e.g., C1, C2, C3)
				expanded_sorted = sorted(expanded, key=lambda x: (len(x), x))
				vec_new.extend(expanded_sorted)
			else:
				vec_new.append(node)
	return vec_new

def discreteness_checker(G, X, Y, obs_data, tell_me_what_discrete = True):
	# Function to check if the columns are binary
	def check_if_binary(obs_data, variables):
		binary_vars = {}
		for var in variables:
			unique_vals = obs_data[var].unique()
			is_binary = set(unique_vals) <= {0, 1}  # Check if unique values are within {0, 1}
			binary_vars[var] = is_binary
		return binary_vars

	# Check if satisfied_BD or satisfied_mSBD holds
	satisfied_BD = adjustment.check_admissibility(G, X, Y)
	satisfied_mSBD = mSBD.constructive_SAC_criterion(G, X, Y)

	if satisfied_BD or satisfied_mSBD:
		# First check if X + Y are binary
		binary_status = check_if_binary(obs_data, X + Y)
		for var, is_var_binary in binary_status.items():
			if not is_var_binary:
				raise ValueError(f'{X+Y} must be discrete (binary). However, {var} is not.')
		
		if tell_me_what_discrete:
			message = f'{X+Y} must be discrete.'
			return (True, message)
		else: 
			return True

	adj_dict_components, adj_dict_operations = identify.return_AC_tree(G, X, Y)
	list_discrete_variables = []

	for adj_dict_component in adj_dict_components.values():
		RootA = adj_dict_component[0]
		PA_RootA = graph.find_parents(G, RootA)
		list_discrete_variables = list_discrete_variables + PA_RootA + RootA 

	list_discrete_variables = list(set(list_discrete_variables))

	# Ensure all relevant variables (including X + Y) are discrete
	must_be_discrete = list(set(list_discrete_variables + X + Y))
	binary_status = check_if_binary(obs_data, must_be_discrete)

	for (var, is_var_binary) in binary_status.items():
		if is_var_binary == False:
			raise ValueError(f'{must_be_discrete} must be discrete (binary). However, {var} is not.')

	if tell_me_what_discrete:
		message = f'{must_be_discrete} must be discrete.'
		return (True, message)
	else: 
		return True

def compute_delta_operation(G, Q_prev_df, S_curr, S_prev):
	"""
	Computes Q_curr using Q_prev, where Q_prev -delta-> Q_curr

	Parameters:
	- Q_prev: dict
		A dictionary where the keys are tuples representing all possible values of S_prev, 
		and the values are the corresponding probabilities.
	- S_curr: list
		A list of variable names representing the subset of S_prev for which we want to compute Q[Si].
	- S_prev: list
		A list of variable names representing the superset that includes all variables in Si.

	Returns:
	- Q_curr: dict
		A dictionary where the keys are tuples representing all possible values of Si, 
		and the values are the computed probabilities Q[Si].
	"""
	# Define topo_V
	topo_V = graph.find_topological_order(G)

	# Initialize Q_curr
	Q_curr = {}

	# Re-sorting the S_curr
	S_curr = sorted(S_curr, key=lambda x: S_prev.index(x))

	S_prev_keys = sorted(list(set(Q_prev_df.keys()) - set(['estimator']) - set(['probability'])), key=lambda x: topo_V.index(x))

	# Domain of S_prev_keys
	domain_S_prev_keys = Q_prev_df[S_prev_keys].drop_duplicates()

	# Iterating over all possible realization of S_prev
	for s_prev_key in domain_S_prev_keys.itertuples(index=False):
		s_prev_key = tuple(s_prev_key)
		Q_curr[s_prev_key] = 1 # initialization 

		for Vi in S_curr:
			# Define variables to be summed for each numerator and denominator 
			summed_variable_numerator = S_prev[S_prev.index(Vi)+1:]
			summed_variable_denominator = S_prev[S_prev.index(Vi):]

			# Domain of summed_variable
			domain_summed_variable_numerator = Q_prev_df[summed_variable_numerator].drop_duplicates()
			domain_summed_variable_denominator = Q_prev_df[summed_variable_denominator].drop_duplicates()

			# Define variables to be fixed for each numerator and denominator 
			fixed_variable_numerator = sorted( list(set(S_prev_keys) - set(summed_variable_numerator)), key=lambda x: topo_V.index(x))
			fixed_variable_denominator = sorted( list(set(S_prev_keys) - set(summed_variable_denominator)), key=lambda x: topo_V.index(x))
			
			# Initialize numerator and denominator 
			numerator = 0
			denominator = 0 

			# Compute Q_curr[(fixed_variable, summed_variable)] += Q_prev[(fixed_variable, summed_variable)]
			# for numerator 
			if len(summed_variable_numerator) == 0:
				numerator = Q_prev_df[(Q_prev_df[S_prev_keys] == s_prev_key).all(axis=1)]['probability'].values[0]
			else:
				for summed_value_numerator in domain_summed_variable_numerator.itertuples(index=False):
					summed_value_numerator = tuple(summed_value_numerator)
					mask_summed = (Q_prev_df[summed_variable_numerator] == summed_value_numerator).all(axis=1)

					fixed_value_numerator = tuple([s_prev_key[S_prev_keys.index(Vi)] for Vi in fixed_variable_numerator])
					mask_fixed = (Q_prev_df[fixed_variable_numerator] == fixed_value_numerator).all(axis=1)

					if (mask_summed & mask_fixed).any():
						numerator += Q_prev_df[(mask_summed) & (mask_fixed)]['probability'].values[0]
					else:
						continue 

			# for denominator  
			if len(fixed_variable_denominator) == 0:
				denominator = 1
			else:
				for summed_value_denominator in domain_summed_variable_denominator.itertuples(index=False):
					summed_value_denominator = tuple(summed_value_denominator)
					mask_summed = (Q_prev_df[summed_variable_denominator] == summed_value_denominator).all(axis=1)

					fixed_value_denominator = tuple([s_prev_key[S_prev_keys.index(Vi)] for Vi in fixed_variable_denominator])
					mask_fixed = (Q_prev_df[fixed_variable_denominator] == fixed_value_denominator).all(axis=1)

					if (mask_summed & mask_fixed).any():
						denominator += Q_prev_df[(mask_summed) & (mask_fixed)]['probability'].values[0]
					else:
						continue 

			if denominator != 0:
				Qi = numerator / denominator
			else:
				Qi = numerator
			Q_curr[s_prev_key] *= Qi

	return Q_curr

def compute_Sigma_operation(G, Q_prev_df, S_curr, S_prev):
	"""
	Computes Q_curr using Q_prev, where Q_prev -delta-> Q_curr

	Parameters:
	- Q_prev: dict
		A dictionary where the keys are tuples representing all possible values of S_prev, 
		and the values are the corresponding probabilities.
	- S_curr: list
		A list of variable names representing the subset of S_prev for which we want to compute Q[Si].
	- S_prev: list
		A list of variable names representing the superset that includes all variables in Si.

	Returns:
	- Q_curr: dict
		A dictionary where the keys are tuples representing all possible values of Si, 
		and the values are the computed probabilities Q[Si].
	"""
	# Define topo_V
	topo_V = graph.find_topological_order(G)

	# Variables in Q_prev_df
	S_prev_keys = sorted( list(set(Q_prev_df.keys()) - set(['estimator']) - set(['probability'])), key=lambda x: topo_V.index(x))
	
	# Variables to be summed
	S_summed = sorted(list(set(S_prev) - set(S_curr)), key=lambda x: S_prev_keys.index(x))

	if len(S_summed) == 0:
		return Q_prev_df 

	# Key variables after marginalization  
	S_diff = sorted(list(set(S_prev_keys) - set(S_summed)), key=lambda x: S_prev_keys.index(x))

	# Initialization 
	Q_diff = {}

	# Domain of S_diff 
	if len(S_diff) > 1:
		domain_S_diff = Q_prev_df[S_diff].drop_duplicates()
	else:
		domain_S_diff = set(np.unique(Q_prev_df[S_summed]))
	
	if len(S_summed) > 1:
		domain_S_summed = Q_prev_df[S_summed].drop_duplicates()
	else:
		domain_S_summed = set(np.unique(Q_prev_df[S_summed]))

	# Iterating over all possible realization of S_prev
	for s_diff in domain_S_diff[S_diff[0]] if len(S_diff) == 1 else domain_S_diff.itertuples(index=False):
		if len(S_diff) > 1:
			s_diff = tuple(s_diff)
		Q_diff[s_diff] = 0 # initialization 
		for s_summed in domain_S_summed:
			mask_summed = (Q_prev_df[S_summed] == s_summed).all(axis=1)
			mask_diff = (Q_prev_df[S_diff] == s_diff).all(axis=1)

			if (mask_summed & mask_diff).any():
				Q_diff[s_diff] += Q_prev_df[(mask_summed) & (mask_diff)]['probability'].values[0]
			else:
				continue 

	return Q_diff

def convert_to_dataframe(data, variable_names, estimator_header=False):
	"""
	Converts a nested dictionary of estimators and variable combinations to a pandas DataFrame.

	Parameters:
	- data: dict
		A dictionary where:
		- If estimator_header is True: The top-level keys are estimator names, and values are dictionaries
		  mapping tuples of variable values to numerical results.
		- If estimator_header is False: A dictionary where keys are tuples of variable values
		  and values are numerical results.
	- variable_names: list
		A list of strings representing the variable names (e.g., ['X', 'Z', 'Y']).
	- estimator_header: bool (default False)
		If True, includes an 'Estimator' column in the DataFrame.

	Returns:
	- pd.DataFrame
		A DataFrame where each row corresponds to a combination of variable values (and estimators if applicable).
	"""

	# Check for the right length of variable names in the second-level keys
	if estimator_header:
		for estimator, results in data.items():
			for key in results.keys():
				if len(key) != len(variable_names):
					raise ValueError(f"Key {key} does not match the length of variable names {variable_names}")
	else:
		for key in data.keys():
			if len(key) != len(variable_names):
				raise ValueError(f"Key {key} does not match the length of variable names {variable_names}")

	# Convert the nested dictionary to a list of dictionaries for easy DataFrame creation
	if estimator_header:
		data_list = [
			{**dict(zip(variable_names, variables)), 'estimator': estimator, 'probability': value}
			for estimator, results in data.items()
			for variables, value in results.items()
		]
	else:
		# Create a list of dictionaries where each entry has variable names and the probability
		data_list = [{**dict(zip(variable_names, key)), 'probability': value} for key, value in data.items()]

	# Create the DataFrame
	df = pd.DataFrame(data_list)

	return df


def estimate_general(G, X, Y, y_val, obs_data, only_OM = False, seednum=123, **kwargs):
	"""
	Estimate the Average Treatment Effect (ATE) using the general framework.

	Parameters:
	G : graph structure representing the causal graph.
	X : list of variables to be conditioned on.
	Y : list of outcome variables.
	y_val : list of values corresponding to Y.
	obs_data : observed data in the form of a DataFrame.
	alpha_CI : confidence level for interval estimates (default is 0.05).
	variance_threshold : threshold for variance estimation (default is 100).
	estimators : method used for estimation (default is "DML").
	seednum : random seed for reproducibility (default is 123).
	
	Returns:
	A dictionary where the keys are tuples of X values and the values are the estimated ATE.
	"""

	if not only_OM:
		raise NotImplementedError("IPW/DML c-component assembly is not implemented; use only_OM=True")

	def get_values(variables, Superset_Values, X, Y, superset_values, x_val, y_val):
		"""
		Get the realized values for the specified variables.

		Parameters:
		variables : list of variables whose values need to be retrieved.
		superset_values : Series containing the values of the superset of variables.
		X : list of variables in X.
		Y : list of variables in Y.
		x_val : list of values corresponding to X.
		y_val : list of values corresponding to Y.

		Returns:
		A list of values for the specified variables.
		"""
		return [
			getattr(superset_values, variable) if variable in Superset_Values else 
			(x_val[X.index(variable)] if variable in X else 
			(y_val[Y.index(variable)] if variable in Y else 1))
			for variable in variables
		]

	def handle_RootA(RootA,  PA_RootA, Superset_Values, X, Y, superset_values, x_val, y_val):
		"""
		Compute the Q[RootA] through mSBD adjustment.

		Parameters:
		RootA : list of variables corresponding to RootA -- RootA is a set of variables s.t. Q[RootA] is mSBD-expressible. 
		Superset_Values : list of all values in the superset of variables.
		superset_values : Series containing the values of the superset of variables.
		x_val : list of values corresponding to X.
		y_val : list of values corresponding to Y.

		Returns:
		Q value for the given RootA.
		"""
		
		# Find the parents of RootA
		if PA_RootA == None:
			PA_RootA = graph.find_parents(G, RootA)

		# Get the values for RootA and its parents
		roota = get_values(variables = RootA, Superset_Values = Superset_Values, X = X, Y = Y, superset_values = superset_values, x_val = x_val, y_val = y_val)
		pa_roota = get_values(variables = PA_RootA, Superset_Values = Superset_Values, X = X, Y = Y, superset_values = superset_values, x_val = x_val, y_val = y_val)

		# Compute the Q value for RootA
		Q_roota, _, _, _  = est_mSBD.estimate_mSBD_xy(G, PA_RootA, RootA, pa_roota, roota, obs_data, only_OM = only_OM, seednum = seednum)
		return Q_roota

	def handle_Next_RootA(RootA, PA_RootA, Next_RootA, Superset_Values, X, Y, superset_values, x_val, y_val):
		"""
		Compute the Q value for a subset of variables (Next_RootA) from Q[RootA].

		Parameters:
		RootA : list of variables corresponding to RootA
		Next_RootA : list of variables corresponding to Next_RootA (subset of RootA)
		Superset_Values : list of all values in the superset of variables.
		superset_values : Series containing the values of the superset of variables.
		x_val : list of values corresponding to X.
		y_val : list of values corresponding to Y.

		Returns:
		Q value for the given Next_RootA.
		"""
		# Find the parents of RootA
		if PA_RootA == None:
			PA_RootA = graph.find_parents(G, RootA)
		
		# Calculate Q[RootA = roota]
		Q_roota = handle_RootA(RootA,  PA_RootA, Superset_Values, X, Y, superset_values, x_val, y_val)

		# Determine RootA - Next_RootA
		RootA_minus_Next = list(set(RootA) - set(Next_RootA))

		# Get the values for RootA and its parents
		roota = get_values(variables = RootA, Superset_Values = Superset_Values, X = X, Y = Y, superset_values = superset_values, x_val = x_val, y_val = y_val)
		pa_roota = get_values(variables = PA_RootA, Superset_Values = Superset_Values, X = X, Y = Y, superset_values = superset_values, x_val = x_val, y_val = y_val)

		Q_roota_next = {}

		# Check if the SAC criterion is satisfied for RootA_minus_Next
		if mSBD.constructive_SAC_criterion(G, PA_RootA, RootA_minus_Next):
			roota_minus_next =  [roota[RootA.index(variable)] for variable in RootA_minus_Next]
			Q_roota_minus_next, _, _, _  = est_mSBD.estimate_mSBD_xy(G, PA_RootA, RootA_minus_Next, pa_roota, roota_minus_next, obs_data, only_OM = only_OM, seednum = seednum) 
			for estimator in list_estimators:
				Q_roota_next[estimator] = min((Q_roota[estimator] / Q_roota_minus_next[estimator]), 1)
			
		# Handle the case where SAC criterion is not satisfied
		else:
			# Sort RootA and RootA_minus_Next according to the topological order
			RootA = sorted(RootA, key=lambda x: topo_V.index(x))
			RootA_minus_Next = sorted(RootA_minus_Next, key=lambda x: RootA.index(x))

			# Initialize Q_roota_next as 1 
			Q_roota_next = {}
			for estimator in list_estimators:
				Q_roota_next[estimator] = 1

			# Q[roota_next] = \prod_{Vj_i \in RootA_minus_Next} (Q_roota_leq_i / Q_roota_less_i)
			for Vj_i in RootA_minus_Next:
				Vj_i_index = RootA.index(Vj_i)
				RootA_leq_i = RootA[:(Vj_i_index+1)]
				roota_leq_i = get_values(variables = RootA_leq_i, Superset_Values = Superset_Values, X = X, Y = Y, superset_values = superset_values, x_val = x_val, y_val = y_val)
				Q_roota_leq_i, _, _, _  = est_mSBD.estimate_mSBD_xy(G, PA_RootA, RootA_leq_i, pa_roota, roota_leq_i, obs_data, only_OM = only_OM, seednum = seednum)

				if Vj_i_index == 0:
					for estimator in list_estimators:
						Q_roota_next[estimator] *= Q_roota_leq_i[estimator]
				else:
					RootA_less_i = RootA[:(Vj_i_index)]
					roota_less_i = get_values(variables = RootA_less_i, Superset_Values = Superset_Values, X = X, Y = Y, superset_values = superset_values, x_val = x_val, y_val = y_val)
					Q_roota_less_i, _, _, _  = est_mSBD.estimate_mSBD_xy(G, PA_RootA, RootA_less_i, pa_roota, roota_less_i, obs_data, only_OM = only_OM, seednum = seednum)
					for estimator in list_estimators:
						Q_roota_next[estimator] *= Q_roota_leq_i[estimator]
					Q_roota_next[estimator] *= min((Q_roota_leq_i[estimator] / Q_roota_less_i[estimator]), 1)

			for estimator in list_estimators:
				Q_roota_next[estimator] = min(Q_roota_next[estimator], 1)

		return Q_roota_next

	cluster_variables = kwargs.get('cluster_variables', None)

	np.random.seed(int(seednum))
	random.seed(int(seednum))

	G0 = copy.copy(G)
	G = graph.unfold_graph_from_data(G, cluster_variables, obs_data)

	list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]

	result = identify.return_AC_tree(G, X, Y)
	if result is None:
		raise NotImplementedError("P(Y|do(X)) for this graph is not supported by the c-component estimator (no AC-tree decomposition)")
	adj_dict_components, adj_dict_operations = result

	# Find V\X and create the corresponding subgraph
	topo_V = graph.find_topological_order(G)
	V_minus_X = list(set(G.nodes()).difference(set(X)))
	subgraph_V_minus_X = graph.subgraphs(G, V_minus_X)

	# Find ancestors of Y in G(V\X)
	D = graph.find_ancestor(subgraph_V_minus_X, Y)
	D_minus_Y = list(set(D) - set(Y))

	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Vi]) for Vi in X]), columns=X)
	ATE = {}
	for estimator in list_estimators:
		ATE[estimator] = {}

	for _, x_val in X_values_combinations.iterrows():
		x_val_tuple = tuple(x_val)
		for estimator in list_estimators:
			ATE[estimator][x_val_tuple] = 0

		for d_minus_y in obs_data[D_minus_Y].drop_duplicates().itertuples(index=False) if len(D_minus_Y) > 0 else [pd.Series()]:
			Q_D_val = {} 
			for estimator in list_estimators:
				Q_D_val[estimator] = 1
		
			for (keyidx, adj_dict_component) in adj_dict_components.items():
				# Case 1. len(adj_dict_component) == 1 (That is, Di = adj_dict_component[0])
				if len(adj_dict_component) == 1:
					Dj = adj_dict_component[0]
					Q_Dj_val = handle_RootA(RootA = Dj, PA_RootA = None, Superset_Values = D_minus_Y, X = X, Y = Y, superset_values = d_minus_y, x_val = x_val, y_val = y_val)
					for estimator in list_estimators:
						Q_D_val[estimator] *= Q_Dj_val[estimator]
					continue

				# Case 2. len(adj_dict_component) == 2 (That is, Di = adj_dict_component[1])
				# elif len(adj_dict_component) == 2: 
				# 	S0 = adj_dict_component[0]
				# 	PA_S0 = graph.find_parents(G, S0)
				# 	Dj = adj_dict_component[1]

				# 	Q_Dj_val = handle_Next_RootA(RootA = S0, PA_RootA = PA_S0, Next_RootA = Dj, Superset_Values = D_minus_Y, X = X, Y = Y, superset_values = d_minus_y, x_val = x_val, y_val = y_val)
				# 	for estimator in list_estimators:
				# 		Q_D_val[estimator] *= Q_Dj_val[estimator]
				# 	continue	

				# Case 3. len(adj_dict_component) > 2 (That is, Di = adj_dict_component[1])
				else:
					'''
					Step 1. Compute Q_S1 
					'''
					adj_dict_component_copy = copy.copy(adj_dict_component)
					adj_dict_operation_copy = copy.copy(adj_dict_operations[keyidx])

					S_prev = adj_dict_component_copy.pop(0)
					S_prev = sorted(S_prev, key=lambda x: topo_V.index(x))
					PA_Sprev = graph.find_parents(G, S_prev)

					# Value of PA_Sprev
					pa_sprev_value = {}
					# Loop through each variable in Dj_variables
					for var in PA_Sprev:
						if var in Y:
							pa_sprev_value[var] = y_val[Y.index(var)]  # Rule 1: Set value as y_val for 'Y'
						elif var in X:
							pa_sprev_value[var] = x_val[X.index(var)]  # Rule 2: Set value as x_val for 'X'
						elif var in D_minus_Y:
							pa_sprev_value[var] = getattr(d_minus_y, var)  # Rule 3: Set value as d_minus_y for 'D_minus_Y'
						else:
							pa_sprev_value[var] = 1 

					domain_Sprev = [tuple(v) for v in obs_data[S_prev].drop_duplicates().itertuples(index=False)]
					
					Q_Sprev = {}
					for estimator in list_estimators:
						Q_Sprev[estimator] = {}

					for s_prev in domain_Sprev:
						Q_Sprev_val, _, _, _ = est_mSBD.estimate_mSBD_xy(G, PA_Sprev, S_prev, list(pa_sprev_value.values()), s_prev, obs_data, only_OM = only_OM, seednum = seednum)
						for estimator in list_estimators:
							Q_Sprev[estimator][s_prev] = Q_Sprev_val[estimator]

					# DataFrame 
					Q_Sprev_df = convert_to_dataframe(Q_Sprev, S_prev, estimator_header = True)

					while adj_dict_component_copy:
						S_curr = adj_dict_component_copy.pop(0)
						S_curr = sorted(S_curr, key=lambda x: S_prev.index(x))

						operator_Scurr = adj_dict_operation_copy.pop(0)
						
						if operator_Scurr == 'δ':
							Q_Scurr = {}
							for estimator in list_estimators:
								Q_Scurr[estimator] = compute_delta_operation(G, Q_Sprev_df[Q_Sprev_df['estimator'] == estimator], S_curr, S_prev)
							keys_Q_Sprev_df = sorted( list(set(Q_Sprev_df.keys()) - set(['estimator']) - set(['probability'])), key=lambda x: topo_V.index(x))
							Q_Scurr_df = convert_to_dataframe(Q_Scurr, keys_Q_Sprev_df, estimator_header = True) 

						elif operator_Scurr == 'Σ':
							for estimator in list_estimators:
								Q_Scurr[estimator] = compute_Sigma_operation(G, Q_Sprev_df[Q_Sprev_df['estimator'] == estimator], S_curr, S_prev)
							keys_Q_Sprev_df = sorted( list(set(Q_Sprev_df.keys()) - set(['estimator']) - set(['probability'])), key=lambda x: topo_V.index(x))
							keys_Q_Scurr_df = sorted( list(set(keys_Q_Sprev_df) - (set(S_prev) - set(S_curr))), key=lambda x: topo_V.index(x))
							Q_Scurr_df = convert_to_dataframe(Q_Scurr, keys_Q_Scurr_df, estimator_header = True) 

						S_prev = copy.copy(S_curr)
						Q_Sprev = copy.copy(Q_Scurr)
						Q_Sprev_df = copy.copy(Q_Scurr_df)

					Q_Dj_df = copy.copy(Q_Scurr_df)
					Q_Dj_val = {}
					
					# Specify the value of dj and pa_dj 
					Dj_variables = list(set(Q_Dj_df.keys()) - set(['estimator']) - set(['probability']))
					dj_value = {}
					# Loop through each variable in Dj_variables
					for var in Dj_variables:
						if var in Y:
							dj_value[var] = y_val[Y.index(var)]  # Rule 1: Set value as y_val for 'Y'
						elif var in X:
							dj_value[var] = x_val[X.index(var)]  # Rule 2: Set value as x_val for 'X'
						elif var in D_minus_Y:
							dj_value[var] = getattr(d_minus_y, var)  # Rule 3: Set value as d_minus_y for 'D_minus_Y'
						else:
							dj_value[var] = 1 

					for estimator in list_estimators:
						filtered_df = Q_Dj_df[Q_Dj_df['estimator'] == estimator]
						
						# Apply filtering dynamically for each key-value pair in dj_value
						for var, value in dj_value.items():
							filtered_df = filtered_df[filtered_df[var] == value]

						# Extract the probability value
						probability_value = filtered_df['probability'].values[0]
						Q_Dj_val[estimator] = probability_value
						Q_D_val[estimator] *= Q_Dj_val[estimator]
					continue

			for estimator in list_estimators:
				ATE[estimator][x_val_tuple] += Q_D_val[estimator]

	return ATE

def estimate_Tian(G, X, Y, y_val, obs_data, only_OM = False, seednum = 123, **kwargs):

	if not only_OM:
		raise NotImplementedError("IPW/DML c-component assembly is not implemented; use only_OM=True")

	cluster_variables = kwargs.get('cluster_variables', None)

	np.random.seed(int(seednum))
	random.seed(int(seednum))

	G0 = copy.copy(G)
	G = graph.unfold_graph_from_data(G, cluster_variables, obs_data)

	topo_V = graph.find_topological_order(G)
	X = sorted(X, key = lambda x: topo_V.index(x))
	SX = sorted( graph.find_c_components(G, X), key=lambda x: topo_V.index(x) )	
	SX_X = sorted( list( set(SX) - set(X) ) , key=lambda x: topo_V.index(x) )	
	V_SX = sorted( list( set(topo_V) - set(SX) ) , key=lambda x: topo_V.index(x) )	
	V_XY = sorted( list( set(topo_V) - set(X + Y)), key=lambda x: topo_V.index(x) )	
	V_Y = sorted( list( set(topo_V) - set(Y)), key=lambda x: topo_V.index(x) )	
	PA_V_SX = graph.find_parents(G, V_SX)
	PA_SX_X = graph.find_parents(G, SX_X)

	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Vi]) for Vi in X]), columns=X)

	list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]
	
	ATE = {}
	for estimator in list_estimators:
		ATE[estimator] = {}

	for _, x_val in X_values_combinations.iterrows():
		for estimator in list_estimators:
			ATE[estimator][tuple(x_val)] = 0
		for v_minus_XY in obs_data[V_XY].drop_duplicates().itertuples(index=False):
			# Compute Q[V\SX](v)
			# di is the realization of Di, defined as follow: For a portion Di \intersect D_minus_Y, take its value from d_minus_y. For Di \setminus D_minus_Y, take the value from y_val.
			v_sx = [
				getattr(v_minus_XY, variable) if variable in V_XY else y_val[Y.index(variable)]
				for variable in V_SX
			]
			# xi is the realization of Xi, defined as follow: For a portion Xi \intersect D_minus_Y, take its value from d_minus_y. For Xi \setminus D_minus_Y, take the value from x_val. Otherwise, just set the value as 1. 
			pa_v_sx = [
				getattr(v_minus_XY, variable) if variable in V_XY else 
				(x_val[X.index(variable)] if variable in X else 1)
				for variable in PA_V_SX
			]
			# Compute Q[V\SX] 
			Q_V_SX_val, _, _, _  = est_mSBD.estimate_mSBD_xy(G, PA_V_SX, V_SX, pa_v_sx, v_sx, obs_data, only_OM = only_OM, seednum = seednum)

			# Compute Q[SX\X] 
			# di is the realization of Di, defined as follow: For a portion Di \intersect D_minus_Y, take its value from d_minus_y. For Di \setminus D_minus_Y, take the value from y_val.
			sx_x = [
				getattr(v_minus_XY, variable) if variable in V_XY else y_val[Y.index(variable)]
				for variable in SX_X
			]
			# xi is the realization of Xi, defined as follow: For a portion Xi \intersect D_minus_Y, take its value from d_minus_y. For Xi \setminus D_minus_Y, take the value from x_val. Otherwise, just set the value as 1. 
			pa_sx_x = [
				getattr(v_minus_XY, variable) if variable in V_XY else 
				(x_val[X.index(variable)] if variable in X else 1)
				for variable in PA_SX_X
			]
			# Compute Q[SX_X] 
			Q_SX_X_val, _, _, _  = est_mSBD.estimate_mSBD_xy(G, PA_SX_X, SX_X, pa_sx_x, sx_x, obs_data, only_OM = only_OM, seednum = seednum)

			for estimator in list_estimators:
				ATE[estimator][tuple(x_val)] += (Q_V_SX_val[estimator] * Q_SX_X_val[estimator])

	return ATE

def estimate_gTian(G, X, Y, y_val, obs_data, only_OM = False, seednum = 123, **kwargs):
	if not only_OM:
		raise NotImplementedError("IPW/DML c-component assembly is not implemented; use only_OM=True")
	cluster_variables = kwargs.get('cluster_variables', None) or []

	np.random.seed(int(seednum))
	random.seed(int(seednum))

	G0 = copy.copy(G)
	G = graph.unfold_graph_from_data(G, cluster_variables, obs_data)

	topo_V = graph.find_topological_order(G)
	X = sorted(X, key = lambda x: topo_V.index(x))
	SX = sorted( graph.find_c_components(G, X), key=lambda x: topo_V.index(x) )	
	SX_X = sorted( list( set(SX) - set(X) ) , key=lambda x: topo_V.index(x) )	
	V_SX = sorted( list( set(topo_V) - set(SX) ) , key=lambda x: topo_V.index(x) )	
	V_XY = sorted( list( set(topo_V) - set(X + Y)), key=lambda x: topo_V.index(x) )	
	V_Y = sorted( list( set(topo_V) - set(Y)), key=lambda x: topo_V.index(x) )	
	PA_V_SX = graph.find_parents(G, V_SX)

	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Vi]) for Vi in X]), columns=X)

	list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]
	
	ATE = {}
	for estimator in list_estimators:
		ATE[estimator] = {}

	idx = 0 
	X_copy = X[:]
	S_Xi_list = [] 
	X_Ci_list = [] 
	
	while len(X_copy) > 0: 
		Xi = X[idx]
		S_Xi = sorted( graph.find_c_components(G, [Xi]), key=lambda x: topo_V.index(x) )
		X_Ci = sorted( list(set(S_Xi).intersection(set(X))), key=lambda x: topo_V.index(x) )
		X_copy = list(set(X_copy) - set(X_Ci))
		idx += 1
		if len(S_Xi) > 1: 
			range_limit = len(S_Xi) if S_Xi[-1] not in X_Ci else len(S_Xi) - next((i for i, x in enumerate(reversed(S_Xi), 1) if not x.startswith('X')), len(S_Xi)) + 1
			if range_limit != len(S_Xi):
				last_X_idx = next((i for i, x in enumerate(reversed(S_Xi), 1) if not x.startswith('X')), len(S_Xi)) - 1
				last_X = S_Xi[-last_X_idx:]
				X_Ci_remained = list(set(X_Ci) - set(last_X))
			else:
				X_Ci_remained = X_Ci[:]
			S_Xi_list.append(S_Xi[:range_limit])
			X_Ci_list.append(X_Ci_remained)

	marginalized_item_list = list(set(V_SX + [item for sublist in S_Xi_list for item in sublist]) - set(X) - set(Y))

	for _, x_val in X_values_combinations.iterrows():
		for estimator in list_estimators:
			ATE[estimator][tuple(x_val)] = 0

		for marginalized_value in obs_data[marginalized_item_list].drop_duplicates().itertuples(index=False):
			Q_VX_val = {}
			for estimator in list_estimators:
				Q_VX_val[estimator] = 1 
			# Compute Q[V\SX](v)
			# di is the realization of Di, defined as follow: For a portion Di \intersect D_minus_Y, take its value from d_minus_y. For Di \setminus D_minus_Y, take the value from y_val.
			v_sx = [
				getattr(marginalized_value, variable) if variable in marginalized_item_list else y_val[Y.index(variable)]
				for variable in V_SX
			]
			# xi is the realization of Xi, defined as follow: For a portion Xi \intersect D_minus_Y, take its value from d_minus_y. For Xi \setminus D_minus_Y, take the value from x_val. Otherwise, just set the value as 1. 
			pa_v_sx = [
				getattr(marginalized_value, variable) if variable in marginalized_item_list else 
				(x_val[X.index(variable)] if variable in X else 1)
				for variable in PA_V_SX
			]
			# Compute Q[V\SX] 
			Q_V_SX_val, _, _, _  = est_mSBD.estimate_mSBD_xy(G, PA_V_SX, V_SX, pa_v_sx, v_sx, obs_data, only_OM = only_OM, seednum = seednum)
			for estimator in list_estimators:
				Q_VX_val[estimator] *= Q_V_SX_val[estimator]

			for idx in range(len(S_Xi_list)):
				SXi = S_Xi_list[idx]
				Xci = X_Ci_list[idx]
				
				# Compute Q[SXi\Xci] 
				SXi_XCi = sorted( list( set(SXi) - set(Xci) ) , key=lambda x: topo_V.index(x) )	
				PA_SXi_XCi = graph.find_parents(G, SXi_XCi)
				# di is the realization of Di, defined as follow: For a portion Di \intersect D_minus_Y, take its value from d_minus_y. For Di \setminus D_minus_Y, take the value from y_val.
				sxi_xci = [
					getattr(marginalized_value, variable) if variable in marginalized_item_list else y_val[Y.index(variable)]
					for variable in SXi_XCi
				]
				# xi is the realization of Xi, defined as follow: For a portion Xi \intersect D_minus_Y, take its value from d_minus_y. For Xi \setminus D_minus_Y, take the value from x_val. Otherwise, just set the value as 1. 
				pa_sx_x = [
					getattr(marginalized_value, variable) if variable in marginalized_item_list else 
					(x_val[X.index(variable)] if variable in X else 1)
					for variable in PA_SXi_XCi
				]
				# Compute Q[SX_X] 
				Q_SXi_Xci_val, _, _, _  = est_mSBD.estimate_mSBD_xy(G, PA_SXi_XCi, SXi_XCi, pa_sx_x, sxi_xci, obs_data, only_OM = only_OM, seednum = seednum)
				for estimator in list_estimators:
					Q_VX_val[estimator] *= Q_SXi_Xci_val[estimator]

			for estimator in list_estimators:
				ATE[estimator][tuple(x_val)] += Q_VX_val[estimator]

	return ATE

def estimate_product_QD(G, X, Y, y_val, obs_data, only_OM = False, seednum = 123):
	if not only_OM:
		raise NotImplementedError("IPW/DML c-component assembly is not implemented; use only_OM=True")
	np.random.seed(int(seednum))
	random.seed(int(seednum))

	result = identify.return_AC_tree(G, X, Y)
	if result is None:
		raise NotImplementedError("P(Y|do(X)) for this graph is not supported by the product-QD c-component estimator (no AC-tree decomposition)")
	adj_dict_components, adj_dict_operations = result
	for adj_dict_component in adj_dict_components.values():
		if len(adj_dict_component) > 1:
			raise ValueError("QD linearity is not satisfied ")

	# Find V\X and create the corresponding subgraph
	V_minus_X = list( set(G.nodes()).difference(set(X)) )
	subgraph_V_minus_X = graph.subgraphs(G,V_minus_X)

	# Find ancestors of Y in G(V\X)
	D = graph.find_ancestor(subgraph_V_minus_X,Y)
	D_minus_Y = list(set(D) - set(Y))

	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Vi]) for Vi in X]), columns=X)

	list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]
	
	ATE = {}
	for estimator in list_estimators:
		ATE[estimator] = {}

	for _, x_val in X_values_combinations.iterrows():
		for estimator in list_estimators:
			ATE[estimator][tuple(x_val)] = 0

		for d_minus_y in obs_data[D_minus_Y].drop_duplicates().itertuples(index=False):
			Q_D_val = {}
			for estimator in list_estimators:
				Q_D_val[estimator] = 1 
			for adj_dict_component in adj_dict_components.values():
				Di = adj_dict_component[-1]
				Xi = graph.find_parents(G, Di)
				
				# di is the realization of Di, defined as follow: For a portion Di \intersect D_minus_Y, take its value from d_minus_y. For Di \setminus D_minus_Y, take the value from y_val.
				di = [
					getattr(d_minus_y, variable) if variable in D_minus_Y else y_val[Y.index(variable)]
					for variable in Di
				]
				# xi is the realization of Xi, defined as follow: For a portion Xi \intersect D_minus_Y, take its value from d_minus_y. For Xi \setminus D_minus_Y, take the value from x_val. Otherwise, just set the value as 1. 
				xi = [
					getattr(d_minus_y, variable) if variable in D_minus_Y else 
					(x_val[X.index(variable)] if variable in X else 1)
					for variable in Xi
				]

				# Compute Q[Di] := P(di | do(xi)) 
				Q_Di_val, _, _, _  = est_mSBD.estimate_mSBD_xy(G, Xi, Di, xi, di, obs_data, only_OM = only_OM, seednum = seednum)

				for estimator in list_estimators:
					Q_D_val[estimator] *= Q_Di_val[estimator]
			
			for estimator in list_estimators:
				ATE[estimator][tuple(x_val)] += Q_D_val[estimator]
	return ATE

def estimate_FD(G, X, Y, y_val, obs_data, only_OM = False, seednum=123, n_folds=2, **kwargs):
	"""
	Front-door estimator with OM / IPW / DML (generalized front-door with baseline
	covariates C, in the style of Fulcher & Tchetgen Tchetgen 2020):

		ψ(x) = Σ_c p(c) Σ_z f(z | x, c) Σ_{x'} E[Y_ind | x', z, c] p(x' | c)

	with Y_ind = 1{Y = y_val}. Nuisances (all cross-fitted, XGBoost):
		f(z | x, c)   - mediator model over discrete Z-configurations
		π(x | c)      - treatment model over discrete X-configurations
		μ(x, z, c)    - outcome model E[Y_ind | x, z, c]
	and with η(z, c) = Σ_{x'} μ(x', z, c) π(x' | c),  ξ(x, c) = Σ_z f(z|x,c) η(z, c):

		OM  = E_n[ ξ(x, C) ]
		IPW = E_n[ Y_ind · f(Z | x, C) / f(Z | X, C) ]
		DML = E_n[ f(Z|x,C)/f(Z|X,C) · (Y_ind − μ(X,Z,C))
		           + 1{X=x}/π(x|C) · (η(Z,C) − ξ(x,C)) + ξ(x,C) ]

	Z and X must be discrete (their observed configurations are enumerated);
	C may be continuous or high-dimensional - it only enters as model features.
	"""
	cluster_variables = kwargs.get('cluster_variables', None)

	np.random.seed(int(seednum))
	random.seed(int(seednum))

	G = graph.unfold_graph_from_data(G, cluster_variables, obs_data)
	ZC = frontdoor.constructive_minimum_FD(G, X, Y)
	if not ZC:
		raise NotImplementedError("estimate_FD: no front-door (Z, C) sets found for this graph")
	Z = sorted(ZC['Z'])
	C = sorted(ZC['C'])

	n = len(obs_data)
	Y_ind = np.ones(n)
	for j, Yj in enumerate(Y):
		Y_ind *= (obs_data[Yj].values == y_val[j]).astype(float)

	# Discrete configurations of Z and X, label-encoded
	z_configs = sorted(set(map(tuple, obs_data[Z].values.tolist())))
	x_configs = sorted(set(map(tuple, obs_data[X].values.tolist())))
	z_code = np.array([z_configs.index(tuple(row)) for row in obs_data[Z].values.tolist()])
	x_code = np.array([x_configs.index(tuple(row)) for row in obs_data[X].values.tolist()])
	nZ, nX = len(z_configs), len(x_configs)

	feat_f  = sorted(set(X) | set(C))          # features for f(z | x, c)
	feat_mu = sorted(set(X) | set(Z) | set(C)) # features for μ(x, z, c)

	def xgb_clf():
		return xgb.XGBClassifier(n_estimators=100, max_depth=4, n_jobs=4, random_state=int(seednum),
								eval_metric='logloss', verbosity=0)
	def xgb_reg():
		return xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_jobs=4,
								booster='gbtree', gamma=0, min_child_weight=1, subsample=0.8,
								alpha=0, random_state=int(seednum), verbosity=0)

	# Cross-fitted per-row nuisance arrays
	F_obs   = np.zeros(n)             # f(Z_i | X_i, C_i)
	F_at    = np.zeros((n, nX, nZ))   # f(z | x, C_i) for every (x, z) configuration
	PI      = np.zeros((n, nX))       # π(x | C_i)
	MU_obs  = np.zeros(n)             # μ(X_i, Z_i, C_i)
	MU_grid = np.zeros((n, nX, nZ))   # μ(x, z, C_i) for every (x, z) configuration

	kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(seednum))
	for train_idx, test_idx in kf.split(obs_data):
		train, test = obs_data.iloc[train_idx], obs_data.iloc[test_idx]

		# f(z-config | X, C)
		f_model = xgb_clf().fit(train[feat_f].values, z_code[train_idx])
		f_classes = list(f_model.classes_)
		proba = f_model.predict_proba(test[feat_f].values)
		F_obs[test_idx] = proba[np.arange(len(test_idx)), [f_classes.index(zc) for zc in z_code[test_idx]]]
		for xi, x_cfg in enumerate(x_configs):
			test_x = test.copy()
			for col, val in zip(X, x_cfg):
				test_x[col] = val
			proba_x = f_model.predict_proba(test_x[feat_f].values)
			for zi in range(nZ):
				F_at[test_idx, xi, zi] = proba_x[:, f_classes.index(zi)] if zi in f_classes else 0.0

		# π(x-config | C)
		if len(C) > 0:
			pi_model = xgb_clf().fit(train[C].values, x_code[train_idx])
			pi_classes = list(pi_model.classes_)
			proba_pi = pi_model.predict_proba(test[C].values)
			for xi in range(nX):
				PI[test_idx, xi] = proba_pi[:, pi_classes.index(xi)] if xi in pi_classes else 0.0
		else:
			for xi in range(nX):
				PI[test_idx, xi] = float(np.mean(x_code[train_idx] == xi))

		# μ(x, z, c) = E[Y_ind | x, z, c]
		mu_model = xgb_reg().fit(train[feat_mu].values, Y_ind[train_idx])
		MU_obs[test_idx] = mu_model.predict(test[feat_mu].values)
		for xi, x_cfg in enumerate(x_configs):
			for zi, z_cfg in enumerate(z_configs):
				test_xz = test.copy()
				for col, val in zip(X, x_cfg):
					test_xz[col] = val
				for col, val in zip(Z, z_cfg):
					test_xz[col] = val
				MU_grid[test_idx, xi, zi] = mu_model.predict(test_xz[feat_mu].values)

	# η(z, C_i) = Σ_{x'} μ(x', z, C_i) π(x' | C_i);   per-row, per-z
	ETA = np.einsum('ixz,ix->iz', MU_grid, PI)
	eta_obs = ETA[np.arange(n), z_code]

	list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]
	ATE = {estimator: {} for estimator in list_estimators}
	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Vi]) for Vi in X]), columns=X)
	for _, x_val in X_values_combinations.iterrows():
		x_tuple = tuple(x_val)
		if x_tuple not in x_configs:
			for estimator in list_estimators:
				ATE[estimator][x_tuple] = 0.0
			continue
		xi = x_configs.index(x_tuple)

		F_x = F_at[:, xi, :]                                   # f(z | x, C_i)
		xi_vals = F_x[np.arange(n), z_code]                    # f(Z_i | x, C_i)
		ksi = np.sum(F_x * ETA, axis=1)                        # ξ(x, C_i)

		ATE["OM"][x_tuple] = float(np.mean(ksi))
		if not only_OM:
			w = xi_vals / np.clip(F_obs, 1e-3, None)
			ATE["IPW"][x_tuple] = float(np.mean(Y_ind * w))
			ind_x = (x_code == xi).astype(float)
			pi_x = np.clip(PI[:, xi], 1e-3, None)
			dml_scores = w * (Y_ind - MU_obs) + (ind_x / pi_x) * (eta_obs - ksi) + ksi
			ATE["DML"][x_tuple] = float(np.mean(dml_scores))
	return ATE

def check_mSBD_ratio(G, X, Y):
	"""
	Check whether P(Y | do(X)) reduces to a single ratio of two mSBD-expressible
	Q-functionals, Q[W*](x, y) / Q[W* \\ Y](x) (e.g., the Napkin graph).

	Returns a dict {"W", "W_minus_C", "R"} describing the ratio, or None.
	"""
	# No outer marginalization: ancestors of Y in G(V \ X) must be exactly Y
	V_minus_X = list(set(G.nodes()).difference(set(X)))
	D = set(graph.find_ancestor(graph.subgraphs(G, V_minus_X), Y)) | set(Y)
	if D != set(Y):
		return None

	result = identify.return_AC_tree(G, X, Y)
	if result is None:
		return None
	adj_dict_components, adj_dict_operations = result
	if len(adj_dict_components) != 1:
		return None
	key = list(adj_dict_components.keys())[0]
	series = adj_dict_components[key]
	operations = adj_dict_operations[key]

	# return_AC_tree gives the clipped series; operations[i] maps series[i] -> series[i+1].
	# The ratio shape is exactly two elements joined by a single δ.
	if len(series) != 2 or operations != ['δ']:
		return None

	topo_V = graph.find_topological_order(G)
	W = sorted(series[0], key=lambda v: topo_V.index(v))
	C = list(series[1])
	R_head = graph.find_parents(G, W)
	if not mSBD.constructive_SAC_criterion(G, R_head, W):
		return None
	# The ratio must isolate exactly Y, and W must be exactly X ∪ Y
	if set(C) != set(Y) or set(W) != set(X) | set(Y):
		return None
	W_minus_C = sorted(set(W).difference(set(C)), key=lambda v: topo_V.index(v))
	R_W = graph.find_parents(G, W)
	R_WC = graph.find_parents(G, W_minus_C)
	if set(R_W) != set(R_WC):
		return None
	if not mSBD.constructive_SAC_criterion(G, R_WC, W_minus_C):
		return None
	return {"W": W, "W_minus_C": W_minus_C, "R": sorted(R_W, key=lambda v: topo_V.index(v))}

def estimate_mSBD_ratio(G, X, Y, y_val, obs_data, only_OM = False, seednum=123, **kwargs):
	"""
	Estimate P(Y | do(X)) for ratio-form c-component effects (e.g., the Napkin):

		P(y | do(x)) = Q[W](x, y; r) / Q[W \\ Y](x; r)      for any value r of R = pa(W),

	where both Q-factors are mSBD-expressible. Each Q-factor is estimated with the
	cross-fitted mSBD engine (OM / IPW / DML via est_mSBD.estimate_mSBD_xy), the
	numerator and denominator are pooled over the observed R-configurations with
	empirical weights P̂(r) (valid since Q[W] = ψ · Q[W \\ Y] for every r), and the
	ratio is taken once at the end. This avoids the per-configuration Q-table
	assembly of estimate_general, which is numerically fragile for IPW/DML.
	"""
	cluster_variables = kwargs.get('cluster_variables', None)

	np.random.seed(int(seednum))
	random.seed(int(seednum))

	G = graph.unfold_graph_from_data(G, cluster_variables, obs_data)

	info = check_mSBD_ratio(G, X, Y)
	if info is None:
		raise NotImplementedError("P(Y|do(X)) for this graph is not a single mSBD ratio; use estimate_general")
	W, W_minus_C, R = info["W"], info["W_minus_C"], info["R"]

	list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]

	# Empirical distribution of the (discrete) parent configurations R
	if len(R) > 0:
		R_counts = obs_data[R].value_counts(normalize=True)
		r_configs = [(tuple(idx) if isinstance(idx, tuple) else (idx,), float(p)) for idx, p in R_counts.items()]
	else:
		r_configs = [((), 1.0)]

	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Vi]) for Vi in X]), columns=X)

	ATE = {estimator: {} for estimator in list_estimators}
	for _, x_val in X_values_combinations.iterrows():
		x_val_tuple = tuple(x_val)
		w_vals = [x_val[v] if v in X else y_val[Y.index(v)] for v in W]
		wc_vals = [x_val[v] for v in W_minus_C]

		num = {estimator: 0.0 for estimator in list_estimators}
		den = {estimator: 0.0 for estimator in list_estimators}
		for r_vals, r_prob in r_configs:
			Q_num, _, _, _ = est_mSBD.estimate_mSBD_xy(G, R, W, list(r_vals), w_vals, obs_data, only_OM = only_OM, seednum = seednum)
			Q_den, _, _, _ = est_mSBD.estimate_mSBD_xy(G, R, W_minus_C, list(r_vals), wc_vals, obs_data, only_OM = only_OM, seednum = seednum)
			for estimator in list_estimators:
				num[estimator] += r_prob * Q_num[estimator]
				den[estimator] += r_prob * Q_den[estimator]

		for estimator in list_estimators:
			ATE[estimator][x_val_tuple] = (num[estimator] / den[estimator]) if den[estimator] > 0 else 0.0
	return ATE

def estimate_case_by_case(G, X, Y, y_val, obs_data, only_OM = False, seednum=123, clip_val = True, minval = 0, maxval = 1, **kwargs):
	# Function to clip ATE values
	def clip_ATE(ATE):
		if clip_val:
			for estimator in ATE.keys():
				for x_val in ATE[estimator].keys():
					ATE[estimator][x_val] = np.clip(ATE[estimator][x_val], minval, maxval)
		return ATE

	cluster_variables = kwargs.get('cluster_variables', None)
	# print(cluster_variables)

	# Stage-2 rewire: normalize cluster info into a cluster_map for the BD/SBD/mSBD estimators.
	cluster_map = kwargs.get('cluster_map', None)
	if cluster_map is None:
		if isinstance(cluster_variables, dict):
			cluster_map = cluster_variables
		elif isinstance(cluster_variables, list):
			cluster_map = graph.build_cluster_map(cluster_variables, obs_data)

	# Check various criteria
	satisfied_BD = adjustment.check_admissibility(G, X, Y)
	# Handle different cases based on criteria
	if satisfied_BD:
		ATE, _, _, _ = est_mSBD.estimate_BD(G, X, Y, obs_data, cluster_map=cluster_map, only_OM=only_OM, seednum=seednum)
		return clip_ATE(ATE)

	satisfied_mSBD = mSBD.constructive_SAC_criterion(G, X, Y)
	if satisfied_mSBD:
		if len(Y) == 1:
			ATE, _, _, _ = est_mSBD.estimate_SBD(G, X, Y, obs_data, cluster_map=cluster_map, only_OM=only_OM, seednum=seednum)
		else:
			ATE, _, _, _ = est_mSBD.estimate_mSBD_y(G, X, Y, y_val, obs_data, cluster_map=cluster_map, only_OM=only_OM, seednum=seednum)
		return clip_ATE(ATE)

	satisfied_FD = frontdoor.constructive_minimum_FD(G, X, Y)
	if satisfied_FD:
		ATE = estimate_FD(G, X, Y, y_val, obs_data, only_OM=only_OM, seednum=seednum, cluster_variables = cluster_variables)
		return clip_ATE(ATE)

	satisfied_Tian = tian.check_Tian_criterion(G, X)
	if satisfied_Tian:
		ATE = estimate_Tian(G, X, Y, y_val, obs_data, only_OM=True, cluster_variables = cluster_variables)
		return clip_ATE(ATE)

	satisfied_gTian = tian.check_Generalized_Tian_criterion(G, X)
	if satisfied_gTian:
		ATE = estimate_gTian(G, X, Y, y_val, obs_data, only_OM=True, cluster_variables = cluster_variables)
		return clip_ATE(ATE)

	satisfied_product = tian.check_product_criterion(G, X, Y)
	if satisfied_product:
		ATE = estimate_product_QD(G, X, Y, y_val, obs_data, only_OM=True)
		return clip_ATE(ATE)

	satisfied_ratio = check_mSBD_ratio(G, X, Y)
	if satisfied_ratio:
		ATE = estimate_mSBD_ratio(G, X, Y, y_val, obs_data, only_OM=only_OM, seednum=seednum, cluster_variables = cluster_variables)
		return clip_ATE(ATE)

	ATE = estimate_general(G, X, Y, y_val, obs_data, only_OM=True, cluster_variables = cluster_variables)
	return clip_ATE(ATE)


if __name__ == "__main__":
	# Generate random SCM and preprocess the graph
	# seednum = int(time.time())
	seednum = 190602

	print(f'Random seed: {seednum}')
	np.random.seed(seednum)
	random.seed(seednum)



	# scm, X, Y = random_generator.Random_SCM_Generator(
	# 	num_observables=7, num_unobservables=2, num_treatments=2, num_outcomes=1,
	# 	condition_ID=True, 
	# 	condition_BD=True, 
	# 	condition_mSBD=True, 
	# 	condition_FD=False, 
	# 	condition_Tian=True, 
	# 	condition_gTian=True,
	# 	condition_product = True, 
	# 	discrete = True, 
	# 	seednum = seednum 
	# )

	# scm, X, Y = example_SCM.BD_SCM(seednum = seednum, d=5); cluster_variables = ['C']
	# scm, X, Y = example_SCM.Kang_Schafer(seednum = seednum)
	# scm, X, Y = example_SCM.Kang_Schafer_dim(seednum = seednum, d = 20); cluster_variables = ['Z']
	# scm, X, Y = example_SCM.Dukes_Vansteelandt_Farrel(seednum = seednum); cluster_variables = ['Z']

	# scm, X, Y = example_SCM.mSBD_SCM(seednum = seednum, d=3); cluster_variables = ['Z1', 'Z2']
	# scm, X, Y = example_SCM.Luedtke_v1(seednum = seednum); cluster_variables = []	
	# scm, X, Y = example_SCM.Luedtke_v2(seednum = seednum); cluster_variables = []	

	scm, X, Y = example_SCM.FD_SCM(seednum = seednum); cluster_variables = ['C', 'Z']
	scm, X, Y = example_SCM.Fulcher_FD(seednum = seednum); cluster_variables = []

	# scm, X, Y = example_SCM.Plan_ID_SCM(seednum = seednum)
	# scm, X, Y = example_SCM.Napkin_SCM(seednum = seednum)
	# scm, X, Y = example_SCM.Napkin_FD_SCM(seednum = seednum)
	# scm, X, Y = example_SCM.Nested_Napkin_SCM(seednum = seednum)
	# scm, X, Y = example_SCM.Double_Napkin_SCM(seednum = seednum)
	# scm, X, Y = example_SCM.Napkin_FD_v2_SCM(seednum = seednum)
	

	G = scm.graph
	G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
	topo_V = graph.find_topological_order(G)

	df_SCM = scm.generate_samples(10000, seed=seednum)
	observables = [node for node in df_SCM.columns if not node.startswith('U')]

	obs_data = df_SCM[observables]

	# Check various criteria
	# satisfied_BD = adjustment.check_admissibility(G, X, Y)
	# satisfied_mSBD = mSBD.constructive_SAC_criterion(G, X, Y)
	# satisfied_FD = frontdoor.constructive_FD(G, X, Y)
	# satisfied_Tian = tian.check_Tian_criterion(G, X)
	# satisfied_gTian = tian.check_Generalized_Tian_criterion(G, X)
	# satisfied_product = tian.check_product_criterion(G, X, Y)

	# print(discreteness_checker(G, X, Y, obs_data))
	print(identify.causal_identification(G, X, Y, True, True))
	# adj_dict_components, adj_dict_operations = identify.return_AC_tree(G, X, Y)

	y_val = np.ones(len(Y)).astype(int)
	truth = statmodules.ground_truth(scm, X, Y, y_val)

	ATE = estimate_case_by_case(G, X, Y, y_val, obs_data, clip_val = False, cluster_variables = cluster_variables)

	performance_table, rank_correlation_table, performance_dict, rank_correlation_dict = statmodules.compute_performance(truth, ATE)
	print("Performance")
	print(performance_table)
	print("Rank Correlation")
	print(rank_correlation_table)

	# # ATE_gen = estimate_general(G, X, Y, y_val, obs_data, only_OM = False)
	# # performance_table_gen, rank_correlation_table_gen = statmodules.compute_performance(truth, ATE_gen)
	# # print(performance_table_gen)

	

	
