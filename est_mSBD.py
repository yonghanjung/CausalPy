import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import KFold
import xgboost as xgb
import copy
import random_generator
import graph
import identify
import adjustment
import frontdoor
import mSBD
import tian
import statmodules

def randomized_equation(**args):
	num_samples = args.pop('num_sample')
	return np.random.binomial(1, 0.5, num_samples)

if __name__ == "__main__":
	# Generate random SCM and preprocess the graph
	scm, X, Y = random_generator.Random_SCM_Generator(
		num_observables=10, num_unobservables=5, num_treatments=4, num_outcomes=1,
		condition_ID=True, condition_BD=False, condition_mSBD=True, 
		condition_FD=False, condition_Tian=True, condition_gTian=True
	)
	G = scm.graph
	G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
	topo_V = graph.find_topological_order(G)
	obs_data = scm.generate_samples(100000)[topo_V]

	print(obs_data)
	print(identify.causal_identification(G, X, Y, False))

	# Check various criteria
	satisfied_BD = adjustment.check_admissibility(G, X, Y)
	satisfied_mSBD = mSBD.constructive_SAC_criterion(G, X, Y)
	satisfied_FD = frontdoor.constructive_FD(G, X, Y)
	satisfied_Tian = tian.check_Tian_criterion(G, X)
	satisfied_gTian = tian.check_Generalized_Tian_criterion(G, X)

	# Assume satisfied_mSBD == True
	if satisfied_mSBD == True:
		print("mSBD holds")

	dict_X, dict_Z, dict_Y = mSBD.check_SAC_with_results(G,X,Y, minimum = True)
	mSBD_length = len(dict_X)

	W_list = []
	for idx in range(mSBD_length):
		W_list.append( dict_Y[f'Y{idx}'] + dict_Z[f'Z{idx+1}'] )
	X_list = list(tuple(dict_X.values()))

	# Update SCM equations with randomized equations for each Xi in X
	for Xi in X:
		scm.equations[Xi] = randomized_equation

	intv_data = scm.generate_samples(1000000)[topo_V]

	causal_effect_estimation = {}
	truth = {}

	# Compute the ground truth for causal effect
	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Xi]) for Xi in X]), columns=X)

	for _, x_val in X_values_combinations.iterrows():
		mask = (intv_data[X] == x_val.values).all(axis=1)
		truth[tuple(x_val)] = intv_data.loc[mask, Y].mean().iloc[0]

	all_Z = []
	for each_Z_list in list(tuple(dict_Z.values())):
		all_Z += each_Z_list

	# Compute causal effect estimations
	if not all_Z:
		for _, x_val in X_values_combinations.iterrows():
			mask = (obs_data[X] == x_val.values).all(axis=1)
			causal_effect_estimation[tuple(x_val)] = obs_data.loc[mask, Y].mean().iloc[0]
	else:
		L = 2
		kf = KFold(n_splits=L, shuffle=True)
		col_feature = list(set(all_Z + X))
		col_label = Y

		# mu3(X3, {Z3, Y2}, X2, {Z2, Y1}, X1, {Z1}) := E[Y3 | X3, Z3, Y2, X2, Z2, Y1, X1, Z1]
		# check_mu3(X2, {Z2, Y1}, X1, {Z1}) := E[Y3 | x3, Z3, Y2, X2, Z2, Y1, X1, Z1]
		# mu2(X2, {Z2, Y1}, X1, {Z1}) := E[check_mu3 | X2, Z2, Y1, X1, Z1]
		# check_mu2(x2, {Z2, Y1}, X1, {Z1}) := E[check_mu3 | x2, Z2, Y1, X1, Z1]
		# mu1(X1, {Z1}) := E[check_mu2 | X1, Z1]
		# check_mu1(X1, {Z1}) := E[check_mu2 | x1, Z1]

		for train_index, test_index in kf.split(obs_data):
			obs_train, obs_test = obs_data.iloc[train_index], obs_data.iloc[test_index]
			mu_m_model = statmodules.learn_mu(obs_train, col_feature, col_label, params=None)
			# mu_XZ = nuisance_mu.predict(xgb.DMatrix(obs_test[col_feature]))
			
			for _, x_val in X_values_combinations.iterrows():
				x_val_list = x_val.values
				for mSBD_idx in reversed(range(mSBD_length)):
					xi = x_val_list[mSBD_idx]
					obs_test_xi = copy.copy(obs_test)
					obs_test_xi[X_list] = x_val.values

					# mu_xZ = mu_m_model.predict(xgb.DMatrix(obs_test_x[col_feature]))
	# 			causal_effect_estimation[tuple(x_val)] = causal_effect_estimation.get(tuple(x_val), 0) + np.mean(mu_xZ)

	# 	for _, x_val in X_values_combinations.iterrows():
	# 		causal_effect_estimation[tuple(x_val)] /= L

	# Evaluate performance
	if causal_effect_estimation:
		performance = np.mean(np.abs(np.array(list(truth.values())) - np.array(list(causal_effect_estimation.values()))))
		print("Performance:", performance)
