import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import KFold
import xgboost as xgb
import copy
from scipy.stats import spearmanr
from scipy.stats import norm
import warnings

import random_generator
import graph
import identify
import adjustment
import frontdoor
import mSBD
import tian
import statmodules

# Turn off alarms
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")

def UCA_decomposition(G, X, topo_V = None):
	# Find the topological order if not provided
	if topo_V is None:
		topo_V = graph.find_topological_order(G)

	# Sort X and find SX, V_SX
	X = sorted(X, key=lambda x: topo_V.index(x))
	SX = sorted(graph.find_c_components(G, X), key=lambda x: topo_V.index(x))
	V_SX = sorted(set(topo_V) - set(SX), key=lambda x: topo_V.index(x))

	appear_V = V_SX[:]

	range_limit = len(SX) if SX[-1] not in X else len(SX) - next((i for i, x in enumerate(reversed(SX), 1) if not x.startswith('X')), len(SX)) + 1
	if range_limit != len(SX):
		last_X_idx = next((i for i, x in enumerate(reversed(SX), 1) if not x.startswith('X')), len(SX)) - 1
		last_X = SX[-last_X_idx:]
		X_remained = list(set(X) - set(last_X))
	else:
		X_remained = X[:]

	for i in range(range_limit):
		Vi = SX[i]
		appear_V.append(Vi)
	
	appear_V = sorted( appear_V, key=lambda x: topo_V.index(x) )
	dict_appear_V = {}
	key_curr = 0
	dict_appear_V[key_curr] = [appear_V[0]]

	if appear_V[0] in SX:
		prev_C = 'SX' 
	else:
		prev_C = 'V_SX' 

	appear_V_pop = appear_V[1:]

	while appear_V_pop:
		pop_node = appear_V_pop.pop(0)
		if pop_node.startswith('Y'):
			dict_appear_V[key_curr+1] = [pop_node]
			break 

		if pop_node in SX: 
			if prev_C == 'SX':
				dict_appear_V[key_curr].append(pop_node)
			else:
				key_curr += 1 
				dict_appear_V[key_curr] = [pop_node]
				prev_C = 'S_X'
		
		elif pop_node in V_SX: 
			if prev_C == 'V_SX':
				dict_appear_V[key_curr].append(pop_node)
			else:
				key_curr += 1 
				dict_appear_V[key_curr] = [pop_node]
				prev_C = 'V_SX'

	dict_predecessor_V = {}
	for idx in range(len(dict_appear_V)):
		dict_predecessor_V[idx] = graph.find_predecessors(G,dict_appear_V[idx])

	
	dict_fixed = {}
	for idx in range(len(dict_appear_V)):
		if dict_appear_V[idx][0] in SX:
			dict_fixed[idx] = []
		else:
			dict_fixed[idx] = []

	return dict_appear_V, dict_predecessor_V

if __name__ == "__main__":
	# Generate random SCM and preprocess the graph
	scm, X, Y = random_generator.Random_SCM_Generator(
		num_observables=10, num_unobservables=5, num_treatments=3, num_outcomes=1,
		condition_ID=True, condition_BD=False, condition_mSBD=False, 
		condition_FD=False, condition_Tian=True, condition_gTian=True
	)
	G = scm.graph
	G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
	topo_V = graph.find_topological_order(G)
	obs_data = scm.generate_samples(10000)[topo_V]

	print(obs_data)
	print(identify.causal_identification(G,X,Y, latex = True, copyTF=False) )
	print(identify.causal_identification(G, X, Y, False, False))

	# Check various criteria
	satisfied_BD = adjustment.check_admissibility(G, X, Y)
	satisfied_mSBD = mSBD.constructive_SAC_criterion(G, X, Y)
	satisfied_FD = frontdoor.constructive_FD(G, X, Y)
	satisfied_Tian = tian.check_Tian_criterion(G, X)
	satisfied_gTian = tian.check_Generalized_Tian_criterion(G, X)

	# Assume satisfied_Tian == True
	dict_appear_V, dict_predecessor_V = UCA_decomposition(G,X)
	appear_V_list = [element for sublist in dict_appear_V.values() for element in sublist]
	m = len(dict_appear_V)-1

	# Update SCM equations with randomized equations for each Xi in X
	for Xi in X:
		scm.equations[Xi] = statmodules.randomized_equation

	intv_data = scm.generate_samples(1000000)[topo_V]

	alpha = 0.05
	z_score = norm.ppf(1 - alpha / 2)
	variance_threshold = 100

	ATE = {}
	VAR = {}
	truth = {}

	# Compute the ground truth for causal effect
	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Xi]) for Xi in X]), columns=X)

	for _, x_val in X_values_combinations.iterrows():
		mask = (intv_data[X] == x_val.values).all(axis=1)
		truth[tuple(x_val)] = intv_data.loc[mask, Y].mean().iloc[0]

	L = 2
	kf = KFold(n_splits=L, shuffle=True)
	col_label = Y

	mu_models = {}
	mu_eval_test_dict = {}
	check_mu_train_dict = {}
	check_mu_test_dict = {}

	pi_eval_dict = {}
	pi_acc_eval_dict = {}

	# Find the topological order if not provided
	if topo_V is None:
		topo_V = graph.find_topological_order(G)

	# Sort X and find SX, V_SX
	X = sorted(X, key=lambda x: topo_V.index(x))
	SX = sorted(graph.find_c_components(G, X), key=lambda x: topo_V.index(x))
	V_SX = sorted(set(topo_V) - set(SX), key=lambda x: topo_V.index(x))

	for Xi in X:
		if Xi in obs_data:
			permuted_values = np.random.permutation(obs_data[Xi])
			obs_data[f'{Xi}perm'] = permuted_values
		else:
			print(f"Warning: {Xi} not found in obs")

	for _, x_val in X_values_combinations.iterrows():
		for train_index, test_index in kf.split(obs_data):
			obs_train, obs_test = obs_data.iloc[train_index], obs_data.iloc[test_index]
			check_mu_train_dict[m+1] = obs_train[Y].values.T[0]
			check_mu_test_dict[m+1] = obs_test[Y].values.T[0]

			for i in range(m, 0, -1):
				Vi = dict_appear_V[i]
				if i == m:
					col_label = Y
				else:
					col_label = f'check_mu_{i+1}'
				
				# col_feature = dict_predecessor_V[i]
				# mu_models[i] = statmodules.learn_mu(obs_train, col_feature, col_label, params=None)
				# mu_eval_test_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_test[col_feature]))


				# if Vi in V_SX: 
					
				# else:

				# 	check_mu2_train_x2_0 = mu2_model.predict(xgb.DMatrix(obs_train_x2_0[col_feature_check_mu2].rename(columns={'X1perm': 'X1'})))



				
				
				# obs_test.loc[:,f'mu_{i}'] = mu_eval_test_dict[i]
				
					

				# if Vi in SX:


				# col_feature = []
				# dict_predecessor_V[i]
				# set(dict_predecessor_V[i]).intersection(set(appear_V_list)).intersection(set(X))
	

