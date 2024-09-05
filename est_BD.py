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

import time
import random 
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

# Suppress all UserWarning messages globally from the osqp package
warnings.filterwarnings("ignore", category=UserWarning, module='osqp')


def estimate_BD(G, X, Y, obs_data, alpha_CI = 0.05, EB_samplesize = 200, EB_boosting = 10, seednum = 123, only_OM = False):
	np.random.seed(int(seednum))
	random.seed(int(seednum))

	topo_V = graph.find_topological_order(G)
	Z = adjustment.construct_minimum_adjustment_set(G, X, Y)
	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Xi]) for Xi in X]), columns=X)

	z_score = norm.ppf(1 - alpha_CI / 2)

	list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]

	ATE = {}
	VAR = {}
	lower_CI = {}
	upper_CI = {}

	for estimator in list_estimators:
		ATE[estimator] = {}
		VAR[estimator] = {}
		lower_CI[estimator] = {}
		upper_CI[estimator] = {}


	# Compute causal effect estimations
	if not Z:
		for estimator in list_estimators:
			for _, x_val in X_values_combinations.iterrows():
				mask = (obs_data[X] == x_val.values).all(axis=1)
				ATE[estimator][tuple(x_val)] = obs_data.loc[mask, Y].mean().iloc[0]
				VAR[estimator][tuple(x_val)] = obs_data.loc[mask, Y].var().iloc[0]
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
				if only_OM: 
					OM_est = np.mean(mu_xZ) 
					ATE["OM"][tuple(x_val)] = ATE["OM"].get(tuple(x_val), 0) + OM_est
					VAR["OM"][tuple(x_val)] = VAR["OM"].get(tuple(x_val), 0) + np.mean( (obs_test['mu_xZ'] - OM_est) ** 2 )

				else:
					if len(obs_test) < EB_samplesize:
						pi_XZ = statmodules.entropy_balancing(obs = obs_test, 
																x_val = x_val.values, 
																X = X, 
																Z = Z, 
																col_feature_1 = 'mu_xZ', 
																col_feature_2 = 'mu_XZ')
					else: 
						pi_XZ = statmodules.entropy_balancing_booster(obs = obs_test, 
																		x_val = x_val.values, 
																		Z = Z, 
																		X = X, 
																		col_feature_1 = 'mu_xZ', 
																		col_feature_2 = 'mu_XZ',
																		B=EB_boosting, 
																		batch_size=EB_samplesize)

					OM_est = np.mean(mu_xZ) 
					ATE["OM"][tuple(x_val)] = ATE["OM"].get(tuple(x_val), 0) + OM_est
					VAR["OM"][tuple(x_val)] = VAR["OM"].get(tuple(x_val), 0) + np.mean( (obs_test['mu_xZ'] - OM_est) ** 2 )

					Yvec = (obs_test[Y].values.flatten())
					PW_est = np.mean(pi_XZ * Yvec)
					ATE["IPW"][tuple(x_val)] = ATE["IPW"].get(tuple(x_val), 0) + PW_est
					VAR["IPW"][tuple(x_val)] = VAR["IPW"].get(tuple(x_val), 0) + np.mean( (pi_XZ * Yvec - PW_est) ** 2 )

					AIPW_est = OM_est + PW_est - np.mean( pi_XZ *mu_XZ )
					AIPW_pseudo_outcome = mu_xZ + pi_XZ * (Yvec - mu_XZ)
					ATE["DML"][tuple(x_val)] = ATE["DML"].get(tuple(x_val), 0) + AIPW_est
					VAR["DML"][tuple(x_val)] = VAR["DML"].get(tuple(x_val), 0) + np.mean( (AIPW_pseudo_outcome - AIPW_est) ** 2 )

		for _, x_val in X_values_combinations.iterrows():
			for estimator in list_estimators:
				ATE[estimator][tuple(x_val)] /= L
				VAR[estimator][tuple(x_val)] /= L

	for _, x_val in X_values_combinations.iterrows():
		for estimator in list_estimators:
			mean_ATE_x = ATE[estimator][tuple(x_val)]
			lower_x = (mean_ATE_x - z_score * VAR[estimator][tuple(x_val)] * (len(obs_data) ** (-1/2)) )
			upper_x = (mean_ATE_x + z_score * VAR[estimator][tuple(x_val)] * (len(obs_data) ** (-1/2)) )
			lower_CI[estimator][tuple(x_val)] = lower_x
			upper_CI[estimator][tuple(x_val)] = upper_x

	return ATE, VAR, lower_CI, upper_CI

def estimate_BD_osqp(G, X, Y, obs_data, alpha_CI = 0.05, EB_samplesize = 200, EB_boosting = 10, seednum = 123, only_OM = False):
	np.random.seed(int(seednum))
	random.seed(int(seednum))

	topo_V = graph.find_topological_order(G)
	Z = adjustment.construct_minimum_adjustment_set(G, X, Y)
	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Xi]) for Xi in X]), columns=X)

	z_score = norm.ppf(1 - alpha_CI / 2)

	list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]

	ATE = {}
	VAR = {}
	lower_CI = {}
	upper_CI = {}

	for estimator in list_estimators:
		ATE[estimator] = {}
		VAR[estimator] = {}
		lower_CI[estimator] = {}
		upper_CI[estimator] = {}


	# Compute causal effect estimations
	if not Z:
		for estimator in list_estimators:
			for _, x_val in X_values_combinations.iterrows():
				mask = (obs_data[X] == x_val.values).all(axis=1)
				ATE[estimator][tuple(x_val)] = obs_data.loc[mask, Y].mean().iloc[0]
				VAR[estimator][tuple(x_val)] = obs_data.loc[mask, Y].var().iloc[0]
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
				if only_OM: 
					OM_est = np.mean(mu_xZ) 
					ATE["OM"][tuple(x_val)] = ATE["OM"].get(tuple(x_val), 0) + OM_est
					VAR["OM"][tuple(x_val)] = VAR["OM"].get(tuple(x_val), 0) + np.mean( (obs_test['mu_xZ'] - OM_est) ** 2 )

				else:
					if len(obs_test) < EB_samplesize:
						pi_XZ = statmodules.entropy_balancing_osqp(obs = obs_test, 
																x_val = x_val.values, 
																X = X, 
																Z = Z, 
																col_feature_1 = 'mu_xZ', 
																col_feature_2 = 'mu_XZ')
					else: 
						pi_XZ = statmodules.entropy_balancing_booster_osqp(obs = obs_test, 
																		x_val = x_val.values, 
																		Z = Z, 
																		X = X, 
																		col_feature_1 = 'mu_xZ', 
																		col_feature_2 = 'mu_XZ',
																		B=EB_boosting, 
																		batch_size=EB_samplesize)

					OM_est = np.mean(mu_xZ) 
					ATE["OM"][tuple(x_val)] = ATE["OM"].get(tuple(x_val), 0) + OM_est
					VAR["OM"][tuple(x_val)] = VAR["OM"].get(tuple(x_val), 0) + np.mean( (obs_test['mu_xZ'] - OM_est) ** 2 )

					Yvec = (obs_test[Y].values.flatten())
					PW_est = np.mean(pi_XZ * Yvec)
					ATE["IPW"][tuple(x_val)] = ATE["IPW"].get(tuple(x_val), 0) + PW_est
					VAR["IPW"][tuple(x_val)] = VAR["IPW"].get(tuple(x_val), 0) + np.mean( (pi_XZ * Yvec - PW_est) ** 2 )

					AIPW_est = OM_est + PW_est - np.mean( pi_XZ *mu_XZ )
					AIPW_pseudo_outcome = mu_xZ + pi_XZ * (Yvec - mu_XZ)
					ATE["DML"][tuple(x_val)] = ATE["DML"].get(tuple(x_val), 0) + AIPW_est
					VAR["DML"][tuple(x_val)] = VAR["DML"].get(tuple(x_val), 0) + np.mean( (AIPW_pseudo_outcome - AIPW_est) ** 2 )

		for _, x_val in X_values_combinations.iterrows():
			for estimator in list_estimators:
				ATE[estimator][tuple(x_val)] /= L
				VAR[estimator][tuple(x_val)] /= L

	for _, x_val in X_values_combinations.iterrows():
		for estimator in list_estimators:
			mean_ATE_x = ATE[estimator][tuple(x_val)]
			lower_x = (mean_ATE_x - z_score * VAR[estimator][tuple(x_val)] * (len(obs_data) ** (-1/2)) )
			upper_x = (mean_ATE_x + z_score * VAR[estimator][tuple(x_val)] * (len(obs_data) ** (-1/2)) )
			lower_CI[estimator][tuple(x_val)] = lower_x
			upper_CI[estimator][tuple(x_val)] = upper_x

	return ATE, VAR, lower_CI, upper_CI

if __name__ == "__main__":
	# Generate random SCM and preprocess the graph
	seednum = int(time.time())
	# seednum = 1725560261

	print(f'Random seed: {seednum}')
	np.random.seed(seednum)
	random.seed(seednum)

	scm, X, Y = random_generator.Random_SCM_Generator(
		num_observables=6, num_unobservables=1, num_treatments=2, num_outcomes=1,
		condition_ID=True, 
		condition_BD=True, 
		condition_mSBD=True, 
		condition_FD=False, 
		condition_Tian=True, 
		condition_gTian=True,
		condition_product = True, 
		discrete = False, 
		seednum = seednum 
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

	y_val = np.ones(len(Y)).astype(int)
	truth = statmodules.ground_truth(scm, obs_data, X, Y, y_val)

	EB_samplesize = 200
	EB_boosting = 5 

	start_time = time.process_time()
	ATE, VAR, lower_CI, upper_CI = estimate_BD(G, X, Y, obs_data, alpha_CI = 0.05, EB_samplesize = EB_samplesize, EB_boosting = EB_boosting, seednum = 123, only_OM = False)
	end_time = time.process_time()
	print(f'Time with SciPy minimizer: {end_time - start_time}')
	performance_table, rank_correlation_table = statmodules.compute_performance(truth, ATE)

	print("Performance")
	print(performance_table)

	print("Rank Correlation")
	print(rank_correlation_table)

	start_time = time.process_time()
	ATE, VAR, lower_CI, upper_CI = estimate_BD_osqp(G, X, Y, obs_data, alpha_CI = 0.05, EB_samplesize = 10000, EB_boosting = 1, seednum = 123, only_OM = False)
	end_time = time.process_time()
	print(f'Time with OSQP: {end_time - start_time}')

	performance_table, rank_correlation_table = statmodules.compute_performance(truth, ATE)

	print("Performance")
	print(performance_table)

	print("Rank Correlation")
	print(rank_correlation_table)

	

	

	


