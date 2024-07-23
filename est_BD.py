import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import KFold
import xgboost as xgb
import copy
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
from scipy.stats import spearmanr

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


if __name__ == "__main__":
	# Generate random SCM and preprocess the graph
	scm, X, Y = random_generator.Random_SCM_Generator(
		num_observables=10, num_unobservables=0, num_treatments=5, num_outcomes=1,
		condition_ID=True, condition_BD=True, condition_mSBD=True, 
		condition_FD=False, condition_Tian=True, condition_gTian=True
	)
	G = scm.graph
	G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
	topo_V = graph.find_topological_order(G)
	obs_data = scm.generate_samples(10000)[topo_V]

	print(obs_data)
	print(identify.causal_identification(G, X, Y, False))

	# Check various criteria
	satisfied_BD = adjustment.check_admissibility(G, X, Y)
	satisfied_mSBD = mSBD.constructive_SAC_criterion(G, X, Y)
	satisfied_FD = frontdoor.constructive_FD(G, X, Y)
	satisfied_Tian = tian.check_Tian_criterion(G, X)
	satisfied_gTian = tian.check_Generalized_Tian_criterion(G, X)

	# Assume satisfied_BD == True
	if satisfied_BD == True:
		Z = adjustment.construct_minimum_adjustment_set(G, X, Y)
		# print(f"{Z} is admisslbe w.r.t. {X} and {Y} in G")

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

	# Compute causal effect estimations
	if not Z:
		for _, x_val in X_values_combinations.iterrows():
			mask = (obs_data[X] == x_val.values).all(axis=1)
			ATE[tuple(x_val)] = obs_data.loc[mask, Y].mean().iloc[0]
			variance_val = obs_data.loc[mask, Y].std().iloc[0]
			VAR[tuple(x_val)] = VAR.get(tuple(x_val), 0) + variance_val
	else:
		L = 2
		kf = KFold(n_splits=L, shuffle=True)
		col_feature = list(set(Z + X))
		col_label = Y

		for train_index, test_index in kf.split(obs_data):
			obs_train, obs_test = obs_data.iloc[train_index], obs_data.iloc[test_index]
			nuisance_mu = statmodules.learn_mu(obs_train, col_feature, col_label, params=None)
			mu_XZ = nuisance_mu.predict(xgb.DMatrix(obs_test[col_feature]))

			for _, x_val in X_values_combinations.iterrows():
				obs_test_x = copy.copy(obs_test)
				obs_test_x[X] = x_val.values
				mu_xZ = nuisance_mu.predict(xgb.DMatrix(obs_test_x[col_feature]))
				mu_XZ = nuisance_mu.predict(xgb.DMatrix(obs_test[col_feature]))
				obs_test.loc[:, 'mu_xZ'] = mu_xZ
				obs_test.loc[:, 'mu_XZ'] = mu_XZ
				if len(obs_test) < 1000:
					pi_XZ = statmodules.entropy_balancing(obs_test, x_val.values, X, Z)
				else: 
					pi_XZ = statmodules.entropy_balancing_booster(obs_test, x_val.values, Z, X, B=5, batch_size=100)

				Yvec = (obs_test[Y].values.flatten())
				OM_est = np.mean(mu_xZ) 
				PW_est = np.mean(pi_XZ * Yvec)
				AIPW_val = OM_est + PW_est - np.mean( pi_XZ *mu_XZ )
				variance_val = np.mean( (mu_xZ + pi_XZ * ( Yvec - mu_XZ) - AIPW_val) ** 2 )

				if variance_val >= variance_threshold:
					ATE[tuple(x_val)] = ATE.get(tuple(x_val), 0) + OM_est
					VAR[tuple(x_val)] = VAR.get(tuple(x_val), 0) + np.mean( (mu_XZ - OM_est) ** 2 )
				else:
					ATE[tuple(x_val)] = ATE.get(tuple(x_val), 0) + AIPW_val
					VAR[tuple(x_val)] = VAR.get(tuple(x_val), 0) + variance_val

		for _, x_val in X_values_combinations.iterrows():
			ATE[tuple(x_val)] /= L
			VAR[tuple(x_val)] /= L

	# Evaluate performance
	performance = np.mean(np.abs(np.array(list(truth.values())) - np.array(list(ATE.values()))))
	print("Performance:", performance)

	lower_CI = {}
	upper_CI = {}
	for _, x_val in X_values_combinations.iterrows():
		mean_ATE_x = ATE[tuple(x_val)]
		lower_x = (mean_ATE_x - z_score * VAR[tuple(x_val)] * (len(obs_data) ** (-1/2)) )
		upper_x = (mean_ATE_x + z_score * VAR[tuple(x_val)] * (len(obs_data) ** (-1/2)) )
		lower_CI[tuple(x_val)] = lower_x
		upper_CI[tuple(x_val)] = upper_x
		print(tuple(x_val), mean_ATE_x, (lower_x, upper_x) )

	rank_correlation, rank_p_values = spearmanr(list(truth.values()), list(ATE.values()))
	print(f"Spearman Rank correlation coefficient: {rank_correlation}")
	print(f"P-value: {rank_p_values}")

	

	

	


