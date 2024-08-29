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

def estimate_product_QD(G, X, Y, y_val, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = "DML", seednum = 123):
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
	ATE = dict()

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

def estimate_Tian(G, X, Y, y_val, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = "DML", seednum = 123, MC_integration_threshold = 10):
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
	ATE = dict()

	unique_rows = obs_data[V_XY].drop_duplicates()	
	unique_row_proportions = obs_data[V_XY].value_counts(normalize=True).reset_index(name='proportion')
	unique_rows_with_proportions = pd.merge(unique_rows, unique_row_proportions, on=V_XY, how='left')

	if len(unique_rows) > MC_integration_threshold:
		unique_rows = unique_rows_with_proportions.sample(n=MC_integration_threshold, random_state = seednum)
		# unique_rows = unique_rows_with_proportions.sample(n=MC_integration_threshold, replace = True, weights = 'proportion', random_state = seednum)

	for _, x_val in X_values_combinations.iterrows():
		ATE[tuple(x_val)] = 0
		for v_minus_XY in unique_rows.itertuples(index=False):
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
			Q_V_SX_val, _, _, _  = est_mSBD.estimate_mSBD_xval_yval(G, PA_V_SX, V_SX, pa_v_sx, v_sx, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = estimators)

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
			Q_SX_X_val, _, _, _  = est_mSBD.estimate_mSBD_xval_yval(G, PA_SX_X, SX_X, pa_sx_x, sx_x, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = estimators)

			ATE[tuple(x_val)] += (Q_V_SX_val * Q_SX_X_val)

	return ATE

def estimate_gTian(G, X, Y, y_val, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = "DML", seednum = 123):
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
	ATE = dict()

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


if __name__ == "__main__":
	# Generate random SCM and preprocess the graph
	# seednum = int(time.time())
	# seednum = 1724359232
	# seednum = 1724437942
	# seednum = 1724442741
	seednum = 1724450481
	np.random.seed(seednum)
	random.seed(seednum)

	scm, X, Y = random_generator.Random_SCM_Generator(
		num_observables=6, num_unobservables=3, num_treatments=2, num_outcomes=1,
		condition_ID=True, 
		condition_BD=False, 
		condition_mSBD=False, 
		condition_FD=True, 
		condition_Tian=True, 
		condition_gTian=True,
		condition_product = True, 
		discrete = True, seednum = seednum 
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
		ATE_OM = estimate_Tian(G, X, Y, y_val, obs_data, alpha_CI = 0.05, variance_threshold = 5, estimators = "OM")
		ATE_DML = estimate_Tian(G, X, Y, y_val, obs_data, alpha_CI = 0.05, variance_threshold = 5, estimators = "DML")
	elif satisfied_product:
		ATE_OM = estimate_product_QD(G, X, Y, y_val, obs_data, alpha_CI = 0.05, variance_threshold = 5, estimators = "OM")
		ATE_DML = estimate_product_QD(G, X, Y, y_val, obs_data, alpha_CI = 0.05, variance_threshold = 5, estimators = "DML")
	elif satisfied_gTian:
		ATE_OM = estimate_gTian(G, X, Y, y_val, obs_data, alpha_CI = 0.05, variance_threshold = 5, estimators = "OM")
		ATE_DML = estimate_gTian(G, X, Y, y_val, obs_data, alpha_CI = 0.05, variance_threshold = 5, estimators = "DML")


	performance_OM = np.mean(np.abs(np.array(list(truth.values())) - np.array(list(ATE_OM.values()))))
	performance_DML = np.mean(np.abs(np.array(list(truth.values())) - np.array(list(ATE_DML.values()))))
	print("Performance (OM):", performance_OM)
	print("Performance (DML):", performance_DML)

	rank_correlation, rank_p_values = spearmanr(list(truth.values()), list(ATE_OM.values()))
	print(f"Spearman Rank correlation coefficient (OM): {rank_correlation}")
	print(f"P-value (OM): {rank_p_values}")

	rank_correlation, rank_p_values = spearmanr(list(truth.values()), list(ATE_DML.values()))
	print(f"Spearman Rank correlation coefficient (DML): {rank_correlation}")
	print(f"P-value: {rank_p_values}")

