import numpy as np 
import pandas as pd 
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy.special import expit
import xgboost as xgb
import copy 

import pyperclip
import graph
import identify
import examples
import random_generator
import SCM 
import adjustment
import frontdoor
import mSBD
import tian 
import statmodules


def randomized_equation(**args):
	num_samples = args.pop('num_sample')
	return np.random.binomial(1, 0.5,num_samples)


if __name__ == "__main__":
	[scm, X, Y] = random_generator.Random_SCM_Generator(num_observables = 10, num_unobservables = 2, num_treatments = 3, num_outcomes = 1, 
																			condition_ID = True, condition_BD = False, condition_mSBD = True, condition_FD = False, condition_Tian = True, condition_gTian = True)
	G = scm.graph
	G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
	topo_V = graph.find_topological_order(G)
	obs_data = scm.generate_samples(100000)[topo_V]
	print(obs_data)
	print( identify.causal_identification(G,X,Y, False) )

	satisfied_BD = adjustment.check_admissibility(G, X, Y)
	satisfied_mSBD = mSBD.constructive_SAC_criterion(G, X, Y)
	satisfied_FD = frontdoor.constructive_FD(G, X, Y)
	satisfied_Tian = tian.check_Tian_criterion(G, X)
	satisfied_gTian = tian.check_Generalized_Tian_criterion(G, X)

	# Assume satisfied_BD == True
	Z = adjustment.construct_minimum_adjustment_set(G, X, Y)

	scm.equations['X1'] = randomized_equation
	intv_data = scm.generate_samples(1000000)[topo_V]

	causal_effect_estimation = {}
	truth = {}

	# Compute the truth 
	if len(X) == 1:
		X_values_combinations = np.unique(np.asarray(obs_data[X]))
		for x_val in X_values_combinations:
			mask = (intv_data[X] == x_val).all(axis=1)
			truth[x_val] = intv_data.loc[mask, Y].mean().iloc[0]
	
	else:
		X_values_combinations = obs_data[X].drop_duplicates()
		for _, x_val in X_values_combinations.iterrows():
			mask = (intv_data[X] == x_val).all(axis=1)
			truth[tuple(x_val)] = intv_data.loc[mask, Y].mean().iloc[0]

	# Compute the the effect
	if len(Z) == 0:
		if len(X) == 1:
			for x_val in X_values_combinations:
				mask = (obs_data[X] == x_val).all(axis=1)
				causal_effect_estimation[x_val] = obs_data.loc[mask, Y].mean().iloc[0]

		else:
			for _, x_val in X_values_combinations.iterrows():
				mask = (obs_data[X] == x_val).all(axis=1)
				causal_effect_estimation[tuple(x_val)] = obs_data.loc[mask, Y].mean().iloc[0]
	
	else:
		# Compute the back-door adjustment module. 
		L = 2 
		kf = KFold(n_splits=L, shuffle=True)
		col_feature = list(set(Z + X))
		col_label = Y 
		for train_index, test_index in kf.split(obs_data):
			obs_train, obs_test = obs_data.iloc[train_index], obs_data.iloc[test_index]
			nuisance_mu = statmodules.learn_mu(obs_train, col_feature, col_label, params = None)
			obs_test_0 = copy.copy(obs_test)
			if len(X) == 1:
				for x_val in X_values_combinations:
					obs_test_x = copy.copy(obs_test)
					obs_test_x[X] = x_val
					pseudo_outcome = nuisance_mu.predict(xgb.DMatrix(obs_test_x[col_feature]))
					causal_effect_estimation[x_val] = np.mean(pseudo_outcome)

			else: 
				for _, x_val in X_values_combinations.iterrows():
					obs_test_x = copy.copy(obs_test)
					obs_test_x[X] = x_val.values
					pseudo_outcome = nuisance_mu.predict(xgb.DMatrix(obs_test_x[col_feature]))
					causal_effect_estimation[tuple(x_val)] = np.mean(pseudo_outcome)


	if len(causal_effect_estimation) > 0:
		performance = np.mean( np.abs( np.array( [v for k,v in truth.items()] ) - np.array( [v for k,v in causal_effect_estimation.items()] ) ) )
		print(performance)






