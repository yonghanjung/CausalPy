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
from tabulate import tabulate


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

def xgb_predict(model, data, col_feature):
	return model.predict(xgb.DMatrix(data[col_feature]))

def estimate_BD(G, X, Y, obs_data, alpha_CI = 0.05, seednum = 123, only_OM = False):
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
			mu_XZ = xgb_predict(nuisance_mu, obs_test, col_feature)

			for _, x_val in X_values_combinations.iterrows():
				obs_test_x = copy.copy(obs_test)
				obs_test_x[X] = x_val.values
				mu_xZ = xgb_predict(nuisance_mu, obs_test_x, col_feature)
				obs_test.loc[:, 'mu_xZ'] = mu_xZ
				obs_test.loc[:, 'mu_XZ'] = mu_XZ
				if only_OM: 
					OM_est = np.mean(mu_xZ) 
					ATE["OM"][tuple(x_val)] = ATE["OM"].get(tuple(x_val), 0) + OM_est
					VAR["OM"][tuple(x_val)] = VAR["OM"].get(tuple(x_val), 0) + np.mean( (obs_test['mu_xZ'] - OM_est) ** 2 )

				else:
					pi_XZ = statmodules.entropy_balancing_osqp(obs = obs_test, 
															x_val = x_val.values, 
															X = X, 
															Z = Z, 
															col_feature_1 = 'mu_xZ', 
															col_feature_2 = 'mu_XZ')

					OM_est = np.mean(mu_xZ) 
					ATE["OM"][tuple(x_val)] = ATE["OM"].get(tuple(x_val), 0) + OM_est
					VAR["OM"][tuple(x_val)] = VAR["OM"].get(tuple(x_val), 0) + np.mean( (obs_test['mu_xZ'] - OM_est) ** 2 )

					Yvec = (obs_test[Y].values.flatten())
					PW_est = np.mean(pi_XZ * Yvec)
					ATE["IPW"][tuple(x_val)] = ATE["IPW"].get(tuple(x_val), 0) + PW_est
					VAR["IPW"][tuple(x_val)] = VAR["IPW"].get(tuple(x_val), 0) + np.mean( (pi_XZ * Yvec - PW_est) ** 2 )

					# AIPW_est = OM_est + PW_est - np.mean( pi_XZ * mu_XZ )
					AIPW_pseudo_outcome = mu_xZ + pi_XZ * (Yvec - mu_XZ)
					AIPW_est = np.mean( AIPW_pseudo_outcome )
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

def estimate_mSBD(G, X, Y, yval, obs_data, alpha_CI = 0.05, seednum = 123, only_OM = False): 
	"""
	Estimate causal effects using the mSBD method.

	Parameters:
	G : Causal graph structure.
	X : List of treatment variables.
	Y : List of outcome variables.
	xval : List of values corresponding to X.
	yval : List of values corresponding to Y.
	obs_data : Observed data (Pandas DataFrame).
	alpha_CI : Confidence level for interval estimates (default 0.05).
	estimators : Method for estimation (default "DML").
	seednum : Random seed for reproducibility (default 123).

	Returns:
	ATE : Estimated average treatment effect.
	VAR : Variance of the estimate.
	lower_CI : Lower confidence interval of the estimate.
	upper_CI : Upper confidence interval of the estimate.
	"""

	np.random.seed(int(seednum))
	random.seed(int(seednum))

	# Sort Y and yval according to the topological order of the graph
	topo_V = graph.find_topological_order(G)
	sorted_pairs = sorted(zip(Y, yval), key=lambda pair: topo_V.index(pair[0]))
	sorted_variables, sorted_values = zip(*sorted_pairs)
	Y = list(sorted_variables)
	yval = list(sorted_values)
	dict_yval = {Y[idx]: yval[idx] for idx in range(len(Y))}
	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Xi]) for Xi in X]), columns=X)

	# Check for SAC criterion satisfaction and organize variables into dictionaries
	dict_X, dict_Z, dict_Y = mSBD.check_SAC_with_results(G,X,Y, minimum = True)
	X_list = list(tuple(dict_X.values()))
	mSBD_length = len(dict_X)

	# Compute IyY: indicator for the outcome variables matching yval
	IyY = ((obs_data[Y] == tuple(yval))*1).prod(axis=1)
	obs_data_y = obs_data[:]
	obs_data_y.loc[:, 'IyY'] = np.asarray(IyY)

	# Create additional indicators for conditional variables
	for idx, (key, value) in enumerate(dict_Y.items()):
		if len(value) > 0:
			list_dict_yval = [dict_yval[value_iter] for value_iter in value]
			obs_data_y.loc[:, f'IyY_{idx}'] = ((obs_data[value] == list_dict_yval).all(axis=1)*1)

	m = len(dict_X)
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

	all_Z = []
	for each_Z_list in list(tuple(dict_Z.values())):
		all_Z += each_Z_list

	# No confounding variables, simple estimation
	if not all_Z:
		for estimator in list_estimators:
			for _, x_val in X_values_combinations.iterrows():
				mask = (obs_data_y[X] == xval).all(axis=1) 
				ATE[estimator] = obs_data_y.loc[mask]['IyY'].mean()
				VAR[estimator] = obs_data_y.loc[mask]['IyY'].var()

	# Confounding variables present, use KFold cross-validation
	else:
		L = 2 # Number of folds 
		kf = KFold(n_splits=L, shuffle=True)

		mu_models = {}
		mu_eval_test_dict = {}
		check_mu_train_dict = {}
		check_mu_test_dict = {}

		pi_eval_dict = {}

		for _, x_val in X_values_combinations.iterrows():
			xval = list(tuple(x_val))
			for train_index, test_index in kf.split(obs_data_y):
				obs_train, obs_test = obs_data_y.iloc[train_index], obs_data_y.iloc[test_index]
				check_mu_train_dict[m+1] = obs_train['IyY'].values
				check_mu_test_dict[m+1] = obs_test['IyY'].values

				# Loop through layers in reverse order
				for i in range(m, 0, -1):
					col_feature = []
					for j in range(1,i+1):
						col_feature += dict_X[f'X{j}']
						col_feature += dict_Z[f'Z{j}']
					for j in range(i):
						col_feature += dict_Y[f'Y{j}']
					col_feature = sorted(col_feature, key=lambda x: topo_V.index(x))
					
					# Label for the current layer
					if i == m:
						col_label = f'IyY_{m}'
					else:
						col_label = f'check_mu_{i+1}'
					
					# Train model for the current layer
					mu_models[i] = statmodules.learn_mu(obs_train, col_feature, col_label, params=None)
					mu_eval_test_dict[i] = xgb_predict(mu_models[i], obs_test, col_feature)
					# mu_eval_test_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_test[col_feature]))
					for j in range(i):
						key_j = f'IyY_{j}'
						if key_j not in obs_data_y: 
							continue 
						else:
							mu_eval_test_dict[i] *= obs_test[key_j]
					obs_test.loc[:,f'mu_{i}'] = mu_eval_test_dict[i]
					
					# Prepare train and test sets for the next iteration
					obs_test_x = copy.copy(obs_test)
					obs_test_x[dict_X[f'X{i}'][0]] = xval[X.index(dict_X[f'X{i}'][0])]
					obs_train_x = copy.copy(obs_train)
					obs_train_x[dict_X[f'X{i}'][0]] = xval[X.index(dict_X[f'X{i}'][0])]

					check_mu_train_dict[i] = xgb_predict(mu_models[i], obs_train_x, col_feature)
					# check_mu_train_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_train_x[col_feature]))
					for j in range(i):
						key_j = f'IyY_{j}'
						if key_j not in obs_data_y: 
							continue 
						else:
							check_mu_train_dict[i] *= obs_train[key_j]
					obs_train.loc[:, f'check_mu_{i}'] = check_mu_train_dict[i]

					check_mu_test_dict[i] = xgb_predict(mu_models[i], obs_test_x, col_feature)
					# check_mu_test_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_test_x[col_feature]))
					for j in range(i):
						key_j = f'IyY_{j}'
						if key_j not in obs_data_y: 
							continue 
						else:
							check_mu_test_dict[i] *= obs_test[key_j]
					obs_test.loc[:, f'check_mu_{i}'] = check_mu_test_dict[i]

					# Compute weights for entropy balancing (if not only outcome model)
					if only_OM == False:
						if i == 1 and len(dict_Y['Y0']) == 0 and len(dict_Z['Z1']) == 0: 
							IxiX = (obs_test[dict_X[f'X{i}'][0]].values == xval[X.index(dict_X[f'X{i}'][0])]) * 1
							P_X1_1 = np.mean(obs_test[dict_X[f'X{i}'][0]].values)
							P_X1 = P_X1_1 * obs_test[dict_X[f'X{i}'][0]].values + (1-P_X1_1) * (1-obs_test[dict_X[f'X{i}'][0]].values )
							pi_XZ = IxiX/P_X1

						else:
							pi_XZ = statmodules.entropy_balancing_osqp(obs = obs_test, 
																	x_val = xval[X.index(dict_X[f'X{i}'][0])], 
																	X = dict_X[f'X{i}'], 
																	Z = list(set(col_feature) - set(dict_X[f'X{i}'])), 
																	col_feature_1 = f'check_mu_{i}', 
																	col_feature_2 = f'mu_{i}')
						
						pi_eval_dict[i] = pi_XZ

				# Outcome model 
				if only_OM:
					OM_val = np.mean(obs_test['check_mu_1'])
					ATE["OM"][tuple(x_val)] = ATE["OM"].get(tuple(x_val), 0) + OM_val
					VAR["OM"][tuple(x_val)] = VAR["OM"].get(tuple(x_val), 0) + np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )

				# Double machine learning (DML), Outcome model (OM) and inverse probability weighting (IPW)
				else:
					pseudo_outcome = np.zeros(len(pi_eval_dict[m]))
					pi_accumulated_dict = {}
					pi_accumulated = np.ones(len(pi_eval_dict[m]))
					for i in range(1,m+1):
						pi_accumulated_dict[i] = pi_accumulated * pi_eval_dict[i]
						pi_accumulated *= pi_eval_dict[i]

					for i in range(m, 0, -1):
						pseudo_outcome += pi_accumulated_dict[i] * (check_mu_test_dict[i+1] - mu_eval_test_dict[i])
					pseudo_outcome += check_mu_test_dict[i]

					OM_val = np.mean(obs_test['check_mu_1'])
					IPW_val = np.mean(pi_accumulated_dict[m] * check_mu_test_dict[m+1])
					AIPW_val = np.mean(pseudo_outcome)

					ATE["OM"][tuple(x_val)] = ATE["OM"].get(tuple(x_val), 0) + OM_val
					VAR["OM"][tuple(x_val)] = VAR["OM"].get(tuple(x_val), 0) + np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )
					
					ATE["DML"][tuple(x_val)] = ATE["DML"].get(tuple(x_val), 0) + AIPW_val
					VAR["DML"][tuple(x_val)] = VAR["DML"].get(tuple(x_val), 0) + np.mean( (pseudo_outcome - AIPW_val) ** 2 )

					ATE["IPW"][tuple(x_val)] = ATE["IPW"].get(tuple(x_val), 0) + IPW_val
					VAR["IPW"][tuple(x_val)] = VAR["IPW"].get(tuple(x_val), 0) + np.mean( (pi_accumulated_dict[m] * check_mu_test_dict[m+1] - IPW_val) ** 2 )
			
			for estimator in list_estimators:
				ATE[estimator][tuple(x_val)] /= L
				VAR[estimator][tuple(x_val)] /= L

			for estimator in list_estimators:
				mean_ATE = ATE[estimator][tuple(x_val)]
				lower_x = (mean_ATE - z_score * VAR[estimator][tuple(x_val)] * (len(obs_data_y) ** (-1/2)) )
				upper_x = (mean_ATE + z_score * VAR[estimator][tuple(x_val)] * (len(obs_data_y) ** (-1/2)) )
				lower_CI[estimator][tuple(x_val)] = lower_x
				upper_CI[estimator][tuple(x_val)] = upper_x
	
	return ATE, VAR, lower_CI, upper_CI

def estimate_mSBD_xval_yval(G, X, Y, xval, yval, obs_data, alpha_CI = 0.05, seednum = 123, only_OM = False): 
	"""
	Estimate causal effects using the mSBD method.

	Parameters:
	G : Causal graph structure.
	X : List of treatment variables.
	Y : List of outcome variables.
	xval : List of values corresponding to X.
	yval : List of values corresponding to Y.
	obs_data : Observed data (Pandas DataFrame).
	alpha_CI : Confidence level for interval estimates (default 0.05).
	estimators : Method for estimation (default "DML").
	seednum : Random seed for reproducibility (default 123).

	Returns:
	ATE : Estimated average treatment effect.
	VAR : Variance of the estimate.
	lower_CI : Lower confidence interval of the estimate.
	upper_CI : Upper confidence interval of the estimate.
	"""

	np.random.seed(int(seednum))
	random.seed(int(seednum))

	# Sort Y and yval according to the topological order of the graph
	topo_V = graph.find_topological_order(G)
	sorted_pairs = sorted(zip(Y, yval), key=lambda pair: topo_V.index(pair[0]))
	sorted_variables, sorted_values = zip(*sorted_pairs)
	Y = list(sorted_variables)
	yval = list(sorted_values)
	dict_yval = {Y[idx]: yval[idx] for idx in range(len(Y))}

	# Check for SAC criterion satisfaction and organize variables into dictionaries
	dict_X, dict_Z, dict_Y = mSBD.check_SAC_with_results(G,X,Y, minimum = True)
	X_list = list(tuple(dict_X.values()))
	mSBD_length = len(dict_X)

	# Compute IyY: indicator for the outcome variables matching yval
	IyY = ((obs_data[Y] == tuple(yval))*1).prod(axis=1)
	obs_data_y = obs_data[:]
	obs_data_y.loc[:, 'IyY'] = np.asarray(IyY)

	# Create additional indicators for conditional variables
	for idx, (key, value) in enumerate(dict_Y.items()):
		if len(value) > 0:
			list_dict_yval = [dict_yval[value_iter] for value_iter in value]
			obs_data_y.loc[:, f'IyY_{idx}'] = ((obs_data[value] == list_dict_yval).all(axis=1)*1)

	m = len(dict_X)
	z_score = norm.ppf(1 - alpha_CI / 2)

	ATE = {}
	VAR = {}
	lower_CI = {}
	upper_CI = {}

	list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]

	for estimator in list_estimators:
		ATE[estimator] = 0
		VAR[estimator] = 0
		lower_CI[estimator] = 0
		upper_CI[estimator] = 0

	all_Z = []
	for each_Z_list in list(tuple(dict_Z.values())):
		all_Z += each_Z_list

	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Xi]) for Xi in X]), columns=X)

	# No confounding variables, simple estimation
	if not all_Z:
		for estimator in list_estimators:
			mask = (obs_data_y[X] == xval).all(axis=1) 
			ATE[estimator] = obs_data_y.loc[mask]['IyY'].mean()
			VAR[estimator] = obs_data_y.loc[mask]['IyY'].var()

	# Confounding variables present, use KFold cross-validation
	else:
		L = 2 # Number of folds 
		kf = KFold(n_splits=L, shuffle=True)

		mu_models = {}
		mu_eval_test_dict = {}
		check_mu_train_dict = {}
		check_mu_test_dict = {}

		pi_eval_dict = {}

		for train_index, test_index in kf.split(obs_data_y):
			obs_train, obs_test = obs_data_y.iloc[train_index], obs_data_y.iloc[test_index]
			check_mu_train_dict[m+1] = obs_train['IyY'].values
			check_mu_test_dict[m+1] = obs_test['IyY'].values

			# Loop through layers in reverse order
			for i in range(m, 0, -1):
				col_feature = []
				for j in range(1,i+1):
					col_feature += dict_X[f'X{j}']
					col_feature += dict_Z[f'Z{j}']
				for j in range(i):
					col_feature += dict_Y[f'Y{j}']
				col_feature = sorted(col_feature, key=lambda x: topo_V.index(x))
				
				# Label for the current layer
				if i == m:
					col_label = f'IyY_{m}'
				else:
					col_label = f'check_mu_{i+1}'
				
				# Train model for the current layer
				mu_models[i] = statmodules.learn_mu(obs_train, col_feature, col_label, params=None)
				mu_eval_test_dict[i] = xgb_predict(mu_models[i], obs_test, col_feature)
				# mu_eval_test_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_test[col_feature]))
				for j in range(i):
					key_j = f'IyY_{j}'
					if key_j not in obs_data_y: 
						continue 
					else:
						mu_eval_test_dict[i] *= obs_test[key_j]
				obs_test.loc[:,f'mu_{i}'] = mu_eval_test_dict[i]
				
				# Prepare train and test sets for the next iteration
				obs_test_x = copy.copy(obs_test)
				obs_test_x[dict_X[f'X{i}'][0]] = xval[X.index(dict_X[f'X{i}'][0])]
				obs_train_x = copy.copy(obs_train)
				obs_train_x[dict_X[f'X{i}'][0]] = xval[X.index(dict_X[f'X{i}'][0])]

				check_mu_train_dict[i] = xgb_predict(mu_models[i], obs_train_x, col_feature)
				# check_mu_train_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_train_x[col_feature]))
				for j in range(i):
					key_j = f'IyY_{j}'
					if key_j not in obs_data_y: 
						continue 
					else:
						check_mu_train_dict[i] *= obs_train[key_j]
				obs_train.loc[:, f'check_mu_{i}'] = check_mu_train_dict[i]

				check_mu_test_dict[i] = xgb_predict(mu_models[i], obs_test_x, col_feature)
				# check_mu_test_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_test_x[col_feature]))
				for j in range(i):
					key_j = f'IyY_{j}'
					if key_j not in obs_data_y: 
						continue 
					else:
						check_mu_test_dict[i] *= obs_test[key_j]
				obs_test.loc[:, f'check_mu_{i}'] = check_mu_test_dict[i]

				# Compute weights for entropy balancing (if not only outcome model)
				if only_OM == False:
					if i == 1 and len(dict_Y['Y0']) == 0 and len(dict_Z['Z1']) == 0: 
						IxiX = (obs_test[dict_X[f'X{i}'][0]].values == xval[X.index(dict_X[f'X{i}'][0])]) * 1
						P_X1_1 = np.mean(obs_test[dict_X[f'X{i}'][0]].values)
						P_X1 = P_X1_1 * obs_test[dict_X[f'X{i}'][0]].values + (1-P_X1_1) * (1-obs_test[dict_X[f'X{i}'][0]].values )
						pi_XZ = IxiX/P_X1

					else:
						pi_XZ = statmodules.entropy_balancing_osqp(obs = obs_test, 
																x_val = xval[X.index(dict_X[f'X{i}'][0])], 
																X = dict_X[f'X{i}'], 
																Z = list(set(col_feature) - set(dict_X[f'X{i}'])), 
																col_feature_1 = f'check_mu_{i}', 
																col_feature_2 = f'mu_{i}')
					
					pi_eval_dict[i] = pi_XZ

			# Outcome model 
			if only_OM:
				OM_val = np.mean(obs_test['check_mu_1'])
				ATE["OM"] += OM_val
				VAR["OM"] += np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )

			# Double machine learning (DML), Outcome model (OM) and inverse probability weighting (IPW)
			else:
				pseudo_outcome = np.zeros(len(pi_eval_dict[m]))
				pi_accumulated_dict = {}
				pi_accumulated = np.ones(len(pi_eval_dict[m]))
				for i in range(1,m+1):
					pi_accumulated_dict[i] = pi_accumulated * pi_eval_dict[i]
					pi_accumulated *= pi_eval_dict[i]

				for i in range(m, 0, -1):
					pseudo_outcome += pi_accumulated_dict[i] * (check_mu_test_dict[i+1] - mu_eval_test_dict[i])
				pseudo_outcome += check_mu_test_dict[i]

				OM_val = np.mean(obs_test['check_mu_1'])
				IPW_val = np.mean(pi_accumulated_dict[m] * check_mu_test_dict[m+1])
				AIPW_val = np.mean(pseudo_outcome)

				ATE["OM"] += OM_val
				VAR["OM"] += np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )
				
				ATE["DML"] += AIPW_val
				VAR["DML"] += np.mean( (pseudo_outcome - AIPW_val) ** 2 )

				ATE["IPW"] += IPW_val
				VAR["IPW"] += np.mean( (pi_accumulated_dict[m] * check_mu_test_dict[m+1] - IPW_val) ** 2 )
		
		for estimator in list_estimators:
			ATE[estimator] /= L
			VAR[estimator] /= L

	for estimator in list_estimators:
		mean_ATE = ATE[estimator]
		lower_x = (mean_ATE - z_score * VAR[estimator] * (len(obs_data_y) ** (-1/2)) )
		upper_x = (mean_ATE + z_score * VAR[estimator] * (len(obs_data_y) ** (-1/2)) )
		lower_CI[estimator] = lower_x
		upper_CI[estimator] = upper_x
	
	return ATE, VAR, lower_CI, upper_CI

def estimate_SBD(G, X, Y, obs_data, alpha_CI = 0.05, seednum = 123, only_OM = False):
	np.random.seed(int(seednum))
	random.seed(int(seednum))

	# Assume satisfied_mSBD == True
	topo_V = graph.find_topological_order(G)
	dict_X, dict_Z, dict_Y = mSBD.check_SAC_with_results(G,X,Y, minimum = True)
	X_list = list(tuple(dict_X.values()))
	mSBD_length = len(dict_X)

	X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Xi]) for Xi in X]), columns=X)

	m = len(X)

	z_score = norm.ppf(1 - alpha_CI / 2)

	ATE = {}
	VAR = {}
	lower_CI = {}
	upper_CI = {}

	list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]

	for estimator in list_estimators:
		ATE[estimator] = {}
		VAR[estimator] = {}
		lower_CI[estimator] = {}
		upper_CI[estimator] = {}

	all_Z = []
	for each_Z_list in list(tuple(dict_Z.values())):
		all_Z += each_Z_list

	# Compute causal effect estimations
	if not all_Z:
		for estimator in list_estimators:
			for _, x_val in X_values_combinations.iterrows():
				mask = (obs_data[X] == x_val.values).all(axis=1)
				ATE[estimator][tuple(x_val)] = obs_data.loc[mask, Y].mean().iloc[0]
				VAR[estimator][tuple(x_val)] = obs_data.loc[mask, Y].var().iloc[0]
	else:
		L = 2
		kf = KFold(n_splits=L, shuffle=True)
		col_feature = list(set(all_Z + X))
		col_label = Y

		mu_models = {}
		mu_eval_test_dict = {}
		check_mu_train_dict = {}
		check_mu_test_dict = {}

		pi_eval_dict = {}
		pi_acc_eval_dict = {}

		for _, x_val in X_values_combinations.iterrows():
			for train_index, test_index in kf.split(obs_data):
				obs_train, obs_test = obs_data.iloc[train_index], obs_data.iloc[test_index]
				check_mu_train_dict[m+1] = obs_train[Y].values.T[0]
				check_mu_test_dict[m+1] = obs_test[Y].values.T[0]

				for i in range(m, 0, -1):
					col_feature = []
					for j in range(1,i+1):
						col_feature += dict_X[f'X{j}']
						col_feature += dict_Z[f'Z{j}']
					col_feature = sorted(col_feature, key=lambda x: topo_V.index(x))
					
					if i == m:
						col_label = Y
					else:
						col_label = f'check_mu_{i+1}'
					
					mu_models[i] = statmodules.learn_mu(obs_train, col_feature, col_label, params=None)
					mu_eval_test_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_test[col_feature]))
					obs_test.loc[:,f'mu_{i}'] = mu_eval_test_dict[i]
					
					obs_test_x = copy.copy(obs_test)
					obs_test_x[dict_X[f'X{i}'][0]] = x_val.values[X.index(dict_X[f'X{i}'][0])]
					obs_train_x = copy.copy(obs_train)
					obs_train_x[dict_X[f'X{i}'][0]] = x_val.values[X.index(dict_X[f'X{i}'][0])]

					check_mu_train_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_train_x[col_feature]))
					obs_train.loc[:, f'check_mu_{i}'] = check_mu_train_dict[i]

					check_mu_test_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_test_x[col_feature]))
					obs_test.loc[:, f'check_mu_{i}'] = check_mu_test_dict[i]

					# If only_OM == False, then the weight should be computed. 
					if only_OM == False:
						if i == 1 and len(dict_Z['Z1']) == 0: 
							IxiX = (obs_test[dict_X[f'X{i}'][0]].values == x_val.values[X.index(dict_X[f'X{i}'][0])]) * 1
							P_X1_1 = np.mean(obs_test[dict_X[f'X{i}'][0]].values)
							P_X1 = P_X1_1 * obs_test[dict_X[f'X{i}'][0]].values + (1-P_X1_1) * (1-obs_test[dict_X[f'X{i}'][0]].values )
							pi_XZ = IxiX/P_X1
							
						else:
							pi_XZ = statmodules.entropy_balancing_osqp(obs = obs_test, 
																	x_val = x_val.values[X.index(dict_X[f'X{i}'][0])], 
																	X = dict_X[f'X{i}'], 
																	Z = list(set(col_feature) - set(dict_X[f'X{i}'])), 
																	col_feature_1 = f'check_mu_{i}', 
																	col_feature_2 = f'mu_{i}')
						pi_eval_dict[i] = pi_XZ

				# If only_OM == True (only returning OM, then no need to compute PW)
				if only_OM:
					OM_val = np.mean(obs_test['check_mu_1'])
					ATE["OM"][tuple(x_val)] = ATE["OM"].get(tuple(x_val), 0) + OM_val
					VAR["OM"][tuple(x_val)] = VAR["OM"].get(tuple(x_val), 0) + np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )

				# If only_OM == False (returning DML, OM, PW)
				else:
					pseudo_outcome = np.zeros(len(pi_eval_dict[m]))
					pi_accumulated_dict = {}
					pi_accumulated = np.ones(len(pi_eval_dict[m]))
					for i in range(1,m+1):
						pi_accumulated_dict[i] = pi_accumulated * pi_eval_dict[i]
						pi_accumulated *= pi_eval_dict[i]

					for i in range(m, 0, -1):
						pseudo_outcome += pi_accumulated_dict[i] * (check_mu_test_dict[i+1] - mu_eval_test_dict[i])
					pseudo_outcome += check_mu_test_dict[i]

					OM_val = np.mean(obs_test['check_mu_1'])
					IPW_val = np.mean(pi_accumulated_dict[m] * check_mu_test_dict[m+1])
					AIPW_val = np.mean(pseudo_outcome)

					ATE["OM"][tuple(x_val)] = ATE["OM"].get(tuple(x_val), 0) + OM_val
					VAR["OM"][tuple(x_val)] = VAR["OM"].get(tuple(x_val), 0) + np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )
					
					ATE["DML"][tuple(x_val)] = ATE["DML"].get(tuple(x_val), 0) + AIPW_val
					VAR["DML"][tuple(x_val)] = VAR["DML"].get(tuple(x_val), 0) + np.mean( (pseudo_outcome - AIPW_val) ** 2 )

					ATE["IPW"][tuple(x_val)] = ATE["IPW"].get(tuple(x_val), 0) + IPW_val
					VAR["IPW"][tuple(x_val)] = VAR["IPW"].get(tuple(x_val), 0) + np.mean( (pi_accumulated_dict[m] * check_mu_test_dict[m+1] - IPW_val) ** 2 )

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

	print(f'Random seed: {seednum}')
	np.random.seed(seednum)
	random.seed(seednum)

	scm, X, Y = random_generator.Random_SCM_Generator(
		num_observables=6, num_unobservables=3, num_treatments=2, num_outcomes=1,
		condition_ID=True, 
		condition_BD=False, 
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

	start_time = time.process_time()
	ATE, VAR, lower_CI, upper_CI = estimate_SBD(G, X, Y, obs_data, alpha_CI = 0.05, seednum = 123, only_OM = False)
	end_time = time.process_time()
	print(f'Time with OSQP minimizer: {end_time - start_time}')

	performance_table, rank_correlation_table = statmodules.compute_performance(truth, ATE)
	
	print("Performance")
	print(performance_table)

	print("Rank Correlation")
	print(rank_correlation_table)

	

