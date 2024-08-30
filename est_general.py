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

import est_mSBD

# Turn off alarms
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")
warnings.simplefilter(action='ignore', category=FutureWarning)

def estimate_general(G, X, Y, y_val, obs_data, only_OM = False, seednum=123, EB_samplesize = 100, EB_boosting = 5):
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

	np.random.seed(int(seednum))
	random.seed(int(seednum))

	list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]

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
		Q_roota, _, _, _  = est_mSBD.estimate_mSBD_xval_yval(G, PA_RootA, RootA, pa_roota, roota, obs_data, only_OM = only_OM, seednum = seednum, EB_samplesize = EB_samplesize, EB_boosting = EB_boosting)
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
			Q_roota_minus_next, _, _, _  = est_mSBD.estimate_mSBD_xval_yval(G, PA_RootA, RootA_minus_Next, pa_roota, roota_minus_next, obs_data, only_OM = only_OM, seednum = seednum, EB_samplesize = EB_samplesize, EB_boosting = EB_boosting) 
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
				Q_roota_leq_i, _, _, _  = est_mSBD.estimate_mSBD_xval_yval(G, PA_RootA, RootA_leq_i, pa_roota, roota_leq_i, obs_data, only_OM = only_OM, seednum = seednum, EB_samplesize = EB_samplesize, EB_boosting = EB_boosting)

				if Vj_i_index == 0:
					for estimator in list_estimators:
						Q_roota_next[estimator] *= Q_roota_leq_i[estimator]
				else:
					RootA_less_i = RootA[:(Vj_i_index)]
					roota_less_i = get_values(variables = RootA_less_i, Superset_Values = Superset_Values, X = X, Y = Y, superset_values = superset_values, x_val = x_val, y_val = y_val)
					Q_roota_less_i, _, _, _  = est_mSBD.estimate_mSBD_xval_yval(G, PA_RootA, RootA_less_i, pa_roota, roota_less_i, obs_data, only_OM = only_OM, seednum = seednum, EB_samplesize = EB_samplesize, EB_boosting = EB_boosting)
					for estimator in list_estimators:
						Q_roota_next[estimator] *= Q_roota_leq_i[estimator]
					Q_roota_next[estimator] *= min((Q_roota_leq_i[estimator] / Q_roota_less_i[estimator]), 1)

			for estimator in list_estimators:
				Q_roota_next[estimator] = min(Q_roota_next[estimator], 1)

		return Q_roota_next

	def compute_QSi_from_QSprev(Q_S_prev, Si, S_prev):
		"""
		Computes Q[Si](si) using Q[S_prev](s_prev), where Si is an arbitrary subset of S_prev.

		Parameters:
		Q_S_prev : dictionary of Q values for S_prev.
		Si : list of variables in the current subset.
		S_prev : list of variables in the previous subset.

		Returns:
		A dictionary of Q values for Si.
		"""

		def get_variable_indices(Si, S_prev):
			"""
			Get the indices of variables in Si within S_prev.
			"""
			return [S_prev.index(var) for var in Si]

		Q_Si = {}

		# Generate all possible realizations of Si
		domain_Si = [tuple(v) for v in obs_data[Si].drop_duplicates().itertuples(index=False)]

		# Get indices of Si's variables in S_prev
		indices_Si = get_variable_indices(Si, S_prev)
		
		for si in domain_Si:
			prob = 1.0
			for j, Vj in enumerate(Si):
				# Summation over the variables in S_prev that are not in Si[j:]
				sum_numerator = 0.0
				sum_denominator = 0.0
				
				for s_prev in Q_S_prev.keys():
					# Projection of s_prev onto the first j+1 variables of Si
					s_prev_proj_upto_j = tuple(s_prev[idx] for idx in indices_Si[:j+1]) # s_prev[si_0, si_1, ..., si_j]
					# The corresponding projection of si
					si_proj_upto_j = si[:j+1] # si_0, ..., si_j

					# Projection of s_prev onto the first j variables of Si
					s_prev_proj_upto_j_minus_1 = tuple(s_prev[idx] for idx in indices_Si[:j]) # s_prev[si_0, si_1, ..., si_j]
					# The corresponding projection of si
					si_proj_upto_j_minus_1 = si[:j] # si_0, ..., si_j

					# Accumulate the sum for the numerator
					if s_prev_proj_upto_j == si_proj_upto_j:
						sum_numerator += Q_S_prev[s_prev] # \sum_{s_prev}Q[S_prev]()
					
					# Accumulate the sum for the denominator
					if s_prev_proj_upto_j_minus_1 == si_proj_upto_j_minus_1:
						sum_denominator += Q_S_prev[s_prev]
				
				if sum_denominator != 0:
					prob *= (sum_numerator / sum_denominator)
			
			Q_Si[si] = prob
		
		return Q_Si

	adj_dict_components, adj_dict_operations = identify.return_AC_tree(G, X, Y)
	
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
			
			for adj_dict_component in adj_dict_components.values():
				# Case 1. len(adj_dict_component) == 1 (That is, Di = adj_dict_component[0])
				if len(adj_dict_component) == 1:
					Dj = adj_dict_component[0]
					Q_Dj_val = handle_RootA(RootA = Dj, PA_RootA = None, Superset_Values = D_minus_Y, X = X, Y = Y, superset_values = d_minus_y, x_val = x_val, y_val = y_val)
					for estimator in list_estimators:
						Q_D_val[estimator] *= Q_Dj_val[estimator]

				# Case 2. len(adj_dict_component) == 2 (That is, Di = adj_dict_component[1])
				elif len(adj_dict_component) == 2: 
					S0 = adj_dict_component[0]
					PA_S0 = graph.find_parents(G, S0)
					Dj = adj_dict_component[1]

					Q_Dj_val = handle_Next_RootA(RootA = S0, PA_RootA = PA_S0, Next_RootA = Dj, Superset_Values = D_minus_Y, X = X, Y = Y, superset_values = d_minus_y, x_val = x_val, y_val = y_val)
					for estimator in list_estimators:
						Q_D_val[estimator] *= Q_Dj_val[estimator]


				# Case 3. len(adj_dict_component) > 2 (That is, Di = adj_dict_component[1])
				else:
					'''
					Step 1. Compute Q_S1 
					'''
					adj_dict_component_copy = copy.copy(adj_dict_component)

					S0 = adj_dict_component_copy.pop(0)
					PA_S0 = graph.find_parents(G, S0)
					domain_S0 = [tuple(v) for v in obs_data[S0].drop_duplicates().itertuples(index=False)]
					
					Q_S0 = {}
					for estimator in list_estimators:
						Q_S0[estimator] = {}

					for s0 in domain_S0:
						Q_s0_val = handle_RootA(RootA = S0, PA_RootA = PA_S0, Superset_Values = S0, X = X, Y = Y, superset_values = pd.Series(s0, S0), x_val = x_val, y_val = y_val)
						for estimator in list_estimators:
							Q_S0[estimator][s0] = Q_s0_val[estimator]

					Q_Sprev = Q_S0
					S_prev = S0

					while adj_dict_component_copy:
						Si = adj_dict_component_copy.pop(0)
						Q_Si = {}
						for estimator in list_estimators:
							Q_Si[estimator] = compute_QSi_from_QSprev(Q_Sprev[estimator], Si, S_prev)

						S_prev = Si 
						Q_Sprev = Q_Si

					Q_Dj_val = {}
					for estimator in list_estimators:
						Q_Dj_val[estimator] = Q_Si[estimator][tuple(get_values(Si, D_minus_Y, X, Y, d_minus_y, x_val, y_val))]
						Q_D_val[estimator] *= Q_Dj_val[estimator]

			for estimator in list_estimators:
				ATE[estimator][x_val_tuple] += Q_D_val[estimator]

	return ATE

def estimate_Tian(G, X, Y, y_val, obs_data, only_OM = False, EB_samplesize = 200, EB_boosting = 5):
	np.random.seed(int(seednum))
	random.seed(int(seednum))

	topo_V = graph.find_topological_order(G)
	X = sorted(X, key = lambda x: topo_V.index(x))
	SX = sorted( graph.find_c_components(G, X), key=lambda x: topo_V.index(x) )	
	SX_X = sorted( list( set(SX) - set(X) ) , key=lambda x: topo_V.index(x) )	
	V_SX = sorted( list( set(topo_V) - set(SX) ) , key=lambda x: topo_V.index(x) )	
	V_XY = sorted( list( set(topo_V) - set(X + Y)), key=lambda x: topo_V.index(x) )	
	V_Y = sorted( list( set(topo_V) - set(Y)), key=lambda x: topo_V.index(x) )	

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
			PA_V_SX = graph.find_parents(G, V_SX)
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
			Q_V_SX_val, _, _, _  = est_mSBD.estimate_mSBD_xval_yval(G, PA_V_SX, V_SX, pa_v_sx, v_sx, obs_data, only_OM = only_OM, seednum = seednum, EB_samplesize = EB_samplesize, EB_boosting = EB_boosting)

			# Compute Q[SX\X] 
			PA_SX_X = graph.find_parents(G, SX_X)
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
			Q_SX_X_val, _, _, _  = est_mSBD.estimate_mSBD_xval_yval(G, PA_SX_X, SX_X, pa_sx_x, sx_x, obs_data, only_OM = only_OM, seednum = seednum, EB_samplesize = EB_samplesize, EB_boosting = EB_boosting)

			for estimator in list_estimators:
				ATE[estimator][tuple(x_val)] += (Q_V_SX_val[estimator] * Q_SX_X_val[estimator])

	return ATE

def estimate_gTian(G, X, Y, y_val, obs_data, only_OM = False, EB_samplesize = 200, EB_boosting = 5):
	np.random.seed(int(seednum))
	random.seed(int(seednum))

	topo_V = graph.find_topological_order(G)
	X = sorted(X, key = lambda x: topo_V.index(x))
	SX = sorted( graph.find_c_components(G, X), key=lambda x: topo_V.index(x) )	
	SX_X = sorted( list( set(SX) - set(X) ) , key=lambda x: topo_V.index(x) )	
	V_SX = sorted( list( set(topo_V) - set(SX) ) , key=lambda x: topo_V.index(x) )	
	V_XY = sorted( list( set(topo_V) - set(X + Y)), key=lambda x: topo_V.index(x) )	
	V_Y = sorted( list( set(topo_V) - set(Y)), key=lambda x: topo_V.index(x) )	

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
		ATE[tuple(x_val)] = 0
		for marginalized_value in obs_data[marginalized_item_list].drop_duplicates().itertuples(index=False):
			Q_VX_val = 1 
			# Compute Q[V\SX](v)
			PA_V_SX = graph.find_parents(G, V_SX)
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
			Q_V_SX_val, _, _, _  = est_mSBD.estimate_mSBD_xval_yval(G, PA_V_SX, V_SX, pa_v_sx, v_sx, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = estimators)
			Q_VX_val *= Q_V_SX_val

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
				Q_SXi_Xci_val, _, _, _  = est_mSBD.estimate_mSBD_xval_yval(G, PA_SXi_XCi, SXi_XCi, pa_sx_x, sxi_xci, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = estimators)
				Q_VX_val *= Q_SXi_Xci_val
			ATE[tuple(x_val)] += Q_VX_val

	return ATE

def estimate_product_QD(G, X, Y, y_val, obs_data, only_OM = False, EB_samplesize = 200, EB_boosting = 5):
	np.random.seed(int(seednum))
	random.seed(int(seednum))

	adj_dict_components, adj_dict_operations = identify.return_AC_tree(G, X, Y)
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
		ATE[tuple(x_val)] = 0
		for d_minus_y in obs_data[D_minus_Y].drop_duplicates().itertuples(index=False):
			Q_D_val = 1 
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
				Q_Di_val, _, _, _  = est_mSBD.estimate_mSBD_xval_yval(G, Xi, Di, xi, di, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = estimators)

				Q_D_val *= Q_Di_val
			
			ATE[tuple(x_val)] += Q_D_val
	return ATE


if __name__ == "__main__":
	# Generate random SCM and preprocess the graph
	seednum = int(time.time())

	print(f'Random seed: {seednum}')
	np.random.seed(seednum)
	random.seed(seednum)

	scm, X, Y = random_generator.Random_SCM_Generator(
		num_observables=6, num_unobservables=3, num_treatments=2, num_outcomes=1,
		condition_ID=True, 
		condition_BD=False, 
		condition_mSBD=False, 
		condition_FD=False, 
		condition_Tian=True, 
		condition_gTian=True,
		condition_product = True, 
		discrete = True, 
		seednum = seednum 
	)

	G = scm.graph
	G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
	topo_V = graph.find_topological_order(G)

	obs_data = scm.generate_samples(2000, seed=seednum)[topo_V]

	# Check various criteria
	satisfied_BD = adjustment.check_admissibility(G, X, Y)
	satisfied_mSBD = mSBD.constructive_SAC_criterion(G, X, Y)
	satisfied_FD = frontdoor.constructive_FD(G, X, Y)
	satisfied_Tian = tian.check_Tian_criterion(G, X)
	satisfied_gTian = tian.check_Generalized_Tian_criterion(G, X)
	satisfied_product = tian.check_product_criterion(G, X, Y)

	print(identify.causal_identification(G, X, Y, False))
	# identify.draw_AC_tree(G,X,Y)

	truth = statmodules.ground_truth(scm, obs_data, X, Y)
	y_val = np.ones(len(Y)).astype(int)

	if satisfied_Tian:
		ATE = estimate_Tian(G, X, Y, y_val, obs_data, only_OM = False)
	elif satisfied_product:
		ATE = estimate_product_QD(G, X, Y, y_val, obs_data, only_OM = False)
	elif satisfied_gTian:
		ATE = estimate_gTian(G, X, Y, y_val, obs_data, only_OM = False)
	else:
		ATE = estimate_general(G, X, Y, y_val, obs_data, only_OM = False)

	performance_table, rank_correlation_table = statmodules.compute_performance(truth, ATE)

	print("Performance")
	print(performance_table)

	print("Rank Correlation")
	print(rank_correlation_table)

	

		# ATE_gen = estimate_general(G, X, Y, y_val, obs_data, only_OM = False)
		# ATE_DML_gen = estimate_general(G, X, Y, y_val, obs_data, alpha_CI = 0.05, variance_threshold = 5, estimators = "DML")

	# performance_OM = np.mean(np.abs(np.array(list(truth.values())) - np.array(list(ATE_OM.values()))))
	# performance_DML = np.mean(np.abs(np.array(list(truth.values())) - np.array(list(ATE_DML.values()))))
	# print("Performance (OM):", performance_OM)
	# print("Performance (DML):", performance_DML)

	# # performance_OM_gen = np.mean(np.abs(np.array(list(truth.values())) - np.array(list(ATE_OM_gen.values()))))
	# # performance_DML_gen = np.mean(np.abs(np.array(list(truth.values())) - np.array(list(ATE_DML_gen.values()))))
	# # print("Performance (OM_gen):", performance_OM_gen)
	# # print("Performance (DML_gen):", performance_DML_gen)

	# rank_correlation, rank_p_values = spearmanr(list(truth.values()), list(ATE_OM.values()))
	# print(f"Spearman Rank correlation coefficient (OM): {rank_correlation}")
	# print(f"P-value (OM): {rank_p_values}")

	# rank_correlation, rank_p_values = spearmanr(list(truth.values()), list(ATE_DML.values()))
	# print(f"Spearman Rank correlation coefficient (DML): {rank_correlation}")
	# print(f"P-value: {rank_p_values}")

