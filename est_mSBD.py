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

# Turn off alarms
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")

def estimate_mSBD_xval_yval(G, X, Y, xval, yval, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = "DML", entropy_balancing_sample_number = 200, B = 10, seednum = 123): 
	np.random.seed(int(seednum))
	random.seed(int(seednum))

	# Assume satisfied_mSBD == True
	topo_V = graph.find_topological_order(G)

	sorted_pairs = sorted(zip(Y, yval), key=lambda pair: topo_V.index(pair[0]))
	sorted_variables, sorted_values = zip(*sorted_pairs)
	Y = list(sorted_variables)
	yval = list(sorted_values)
	dict_yval = {Y[idx]: yval[idx] for idx in range(len(Y))}

	dict_X, dict_Z, dict_Y = mSBD.check_SAC_with_results(G,X,Y, minimum = True)
	X_list = list(tuple(dict_X.values()))
	mSBD_length = len(dict_X)

	# Compute IyY 
	IyY = ((obs_data[Y] == tuple(yval))*1).prod(axis=1)
	obs_data_y = obs_data[:]
	obs_data_y.loc[:, 'IyY'] = np.asarray(IyY)

	for idx, (key, value) in enumerate(dict_Y.items()):
		if len(value) > 0:
			list_dict_yval = [dict_yval[value_iter] for value_iter in value]
			obs_data_y.loc[:, f'IyY_{idx}'] = ((obs_data[value] == list_dict_yval).all(axis=1)*1)

	m = len(dict_X)

	z_score = norm.ppf(1 - alpha_CI / 2)

	all_Z = []
	for each_Z_list in list(tuple(dict_Z.values())):
		all_Z += each_Z_list

	# Compute causal effect estimations
	if not all_Z:
		# for _, x_val in X_values_combinations.iterrows():
		mask = (obs_data_y[X] == xval).all(axis=1) 
		ATE = obs_data_y.loc[mask]['IyY'].mean()
		VAR = obs_data_y.loc[mask]['IyY'].var()
	else:
		ATE = 0
		VAR = 0
		L = 2

		kf = KFold(n_splits=L, shuffle=True)

		mu_models = {}
		mu_eval_test_dict = {}
		check_mu_train_dict = {}
		check_mu_test_dict = {}

		pi_eval_dict = {}
		pi_acc_eval_dict = {}

		for train_index, test_index in kf.split(obs_data_y):
			obs_train, obs_test = obs_data_y.iloc[train_index], obs_data_y.iloc[test_index]
			check_mu_train_dict[m+1] = obs_train['IyY'].values
			check_mu_test_dict[m+1] = obs_test['IyY'].values

			for i in range(m, 0, -1):
				col_feature = []
				for j in range(1,i+1):
					col_feature += dict_X[f'X{j}']
					col_feature += dict_Z[f'Z{j}']
				for j in range(i):
					col_feature += dict_Y[f'Y{j}']
				col_feature = sorted(col_feature, key=lambda x: topo_V.index(x))
				
				if i == m:
					col_label = f'IyY_{m}'
				else:
					col_label = f'check_mu_{i+1}'
				
				mu_models[i] = statmodules.learn_mu(obs_train, col_feature, col_label, params=None)
				mu_eval_test_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_test[col_feature]))
				for j in range(i):
					key_j = f'IyY_{j}'
					if key_j not in obs_data_y: 
						continue 
					else:
						mu_eval_test_dict[i] *= obs_test[key_j]
				obs_test.loc[:,f'mu_{i}'] = mu_eval_test_dict[i]
				
				obs_test_x = copy.copy(obs_test)
				obs_test_x[dict_X[f'X{i}'][0]] = xval[X.index(dict_X[f'X{i}'][0])]
				obs_train_x = copy.copy(obs_train)
				obs_train_x[dict_X[f'X{i}'][0]] = xval[X.index(dict_X[f'X{i}'][0])]

				check_mu_train_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_train_x[col_feature]))
				for j in range(i):
					key_j = f'IyY_{j}'
					if key_j not in obs_data_y: 
						continue 
					else:
						check_mu_train_dict[i] *= obs_train[key_j]
				obs_train.loc[:, f'check_mu_{i}'] = check_mu_train_dict[i]

				check_mu_test_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_test_x[col_feature]))
				for j in range(i):
					key_j = f'IyY_{j}'
					if key_j not in obs_data_y: 
						continue 
					else:
						check_mu_test_dict[i] *= obs_test[key_j]
				obs_test.loc[:, f'check_mu_{i}'] = check_mu_test_dict[i]

				if estimators != "OM":
					# if len(dict_Z[f'Z{i}']) > 0 and i > 1:
					if i == 1 and len(dict_Y['Y0']) == 0 and len(dict_Z['Z1']) == 0: 
						IxiX = (obs_test[dict_X[f'X{i}'][0]].values == xval[X.index(dict_X[f'X{i}'][0])]) * 1
						P_X1_1 = np.mean(obs_test[dict_X[f'X{i}'][0]].values)
						P_X1 = P_X1_1 * obs_test[dict_X[f'X{i}'][0]].values + (1-P_X1_1) * (1-obs_test[dict_X[f'X{i}'][0]].values )
						pi_XZ = IxiX/P_X1

					else:
						if len(obs_test) < entropy_balancing_sample_number:
							pi_XZ = statmodules.entropy_balancing(obs = obs_test, 
																	x_val = xval[X.index(dict_X[f'X{i}'][0])], 
																	X = dict_X[f'X{i}'], 
																	Z = list(set(col_feature) - set(dict_X[f'X{i}'])), 
																	col_feature_1 = f'check_mu_{i}', 
																	col_feature_2 = f'mu_{i}')
						else: 
							pi_XZ = statmodules.entropy_balancing_booster(obs = obs_test, 
																			x_val = xval[X.index(dict_X[f'X{i}'][0])], 
																			Z = list(set(col_feature) - set(dict_X[f'X{i}'])), 
																			X = dict_X[f'X{i}'], 
																			col_feature_1 = f'check_mu_{i}', 
																			col_feature_2 = f'mu_{i}', 
																			B=10, 
																			batch_size=100)
					
					pi_eval_dict[i] = pi_XZ

			if estimators == "OM":
				OM_val = np.mean(obs_test['check_mu_1'])
				ATE += OM_val
				VAR += np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )

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
				variance_val = np.mean( (pseudo_outcome - AIPW_val) ** 2 )

				if variance_val >= variance_threshold:
					ATE += OM_val
					VAR += np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )
				else:
					ATE += AIPW_val
					VAR += variance_val
		
		ATE /= L
		VAR /= L

	lower_CI = {}
	upper_CI = {}

	mean_ATE = ATE
	lower_x = (mean_ATE - z_score * VAR * (len(obs_data_y) ** (-1/2)) )
	upper_x = (mean_ATE + z_score * VAR * (len(obs_data_y) ** (-1/2)) )
	lower_CI = lower_x
	upper_CI = upper_x
	
	return ATE, VAR, lower_CI, upper_CI

def estimate_mSBD(G, X, Y, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = "DML", seednum = 123):
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

	all_Z = []
	for each_Z_list in list(tuple(dict_Z.values())):
		all_Z += each_Z_list

	# Compute causal effect estimations
	if not all_Z:
		for _, x_val in X_values_combinations.iterrows():
			mask = (obs_data[X] == x_val.values).all(axis=1)
			ATE[tuple(x_val)] = obs_data.loc[mask, Y].mean().iloc[0]
			VAR[tuple(x_val)] = obs_data.loc[mask, Y].var().iloc[0]
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

					if estimators != "OM":
						if len(dict_Z[f'Z{i}']) > 0:
							if len(obs_test) < 500:
								pi_XZ = statmodules.entropy_balancing(obs = obs_test, 
																		x_val = x_val.values[X.index(dict_X[f'X{i}'][0])], 
																		X = dict_X[f'X{i}'], 
																		Z = list(set(col_feature) - set(dict_X[f'X{i}'])), 
																		col_feature_1 = f'check_mu_{i}', 
																		col_feature_2 = f'mu_{i}')
							else: 
								pi_XZ = statmodules.entropy_balancing_booster(obs = obs_test, 
																				x_val = x_val.values[X.index(dict_X[f'X{i}'][0])], 
																				Z = list(set(col_feature) - set(dict_X[f'X{i}'])), 
																				X = dict_X[f'X{i}'], 
																				col_feature_1 = f'check_mu_{i}', 
																				col_feature_2 = f'mu_{i}', 
																				B=2, 
																				batch_size=100)
						else:
							IxiX = (obs_test[dict_X[f'X{i}'][0]].values == x_val.values[X.index(dict_X[f'X{i}'][0])]) * 1
							P_X1_1 = np.mean(obs_test[dict_X[f'X{i}'][0]].values)
							P_X1 = P_X1_1 * obs_test[dict_X[f'X{i}'][0]].values + (1-P_X1_1) * (1-obs_test[dict_X[f'X{i}'][0]].values )
							pi_XZ = IxiX/P_X1
						pi_eval_dict[i] = pi_XZ

				if estimators == "OM":
					OM_val = np.mean(obs_test['check_mu_1'])
					ATE[tuple(x_val)] = ATE.get(tuple(x_val), 0) + OM_val
					VAR[tuple(x_val)] = VAR.get(tuple(x_val), 0) + np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )

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
					variance_val = np.mean( (pseudo_outcome - AIPW_val) ** 2 )

					if variance_val >= variance_threshold:
						ATE[tuple(x_val)] = ATE.get(tuple(x_val), 0) + OM_val
						VAR[tuple(x_val)] = VAR.get(tuple(x_val), 0) + np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )
					else:
						ATE[tuple(x_val)] = ATE.get(tuple(x_val), 0) + AIPW_val
						VAR[tuple(x_val)] = VAR.get(tuple(x_val), 0) + variance_val
		
		for _, x_val in X_values_combinations.iterrows():
			ATE[tuple(x_val)] /= L
			VAR[tuple(x_val)] /= L

	lower_CI = {}
	upper_CI = {}

	for _, x_val in X_values_combinations.iterrows():
		mean_ATE_x = ATE[tuple(x_val)]
		lower_x = (mean_ATE_x - z_score * VAR[tuple(x_val)] * (len(obs_data) ** (-1/2)) )
		upper_x = (mean_ATE_x + z_score * VAR[tuple(x_val)] * (len(obs_data) ** (-1/2)) )
		lower_CI[tuple(x_val)] = lower_x
		upper_CI[tuple(x_val)] = upper_x
	
	return ATE, VAR, lower_CI, upper_CI

if __name__ == "__main__":
	# Generate random SCM and preprocess the graph
	# seednum = int(time.time())
	seednum = 1724356263
	np.random.seed(seednum)
	random.seed(seednum)

	scm, X, Y = random_generator.Random_SCM_Generator(
		num_observables=6, num_unobservables=3, num_treatments=2, num_outcomes=1,
		condition_ID=True, condition_BD=False, condition_mSBD=True, 
		condition_FD=False, condition_Tian=True, condition_gTian=True,
		discrete = False, seednum = seednum 
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

	truth = statmodules.ground_truth(scm, obs_data, X, Y)
	
	ATE_DML, VAR_DML, lower_CI_DML, upper_CI_DML = estimate_mSBD(G, X, Y, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = "DML")
	ATE_OM, VAR_OM, lower_CI_OM, upper_CI_OM = estimate_mSBD(G, X, Y, obs_data, alpha_CI = 0.05, variance_threshold = 100, estimators = "OM")

	# Evaluate performance
	performance_DML = np.mean(np.abs(np.array(list(truth.values())) - np.array(list(ATE_DML.values()))))
	performance_OM = np.mean(np.abs(np.array(list(truth.values())) - np.array(list(ATE_OM.values()))))
	print("Performance:", performance_DML, performance_OM)

	# rank_correlation, rank_p_values = spearmanr(list(truth.values()), list(ATE.values()))
	# print(f"Spearman Rank correlation coefficient: {rank_correlation}")
	# print(f"P-value: {rank_p_values}")

