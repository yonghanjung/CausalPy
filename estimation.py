import networkx as nx
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import random
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy 

import graph
import adjustment    
import example_SCM
import mSBD

def value_in_min_max(val, minval=0.01, maxval=0.99):
	return np.min( [np.max([val,minval]), maxval] )

def clip_vector(vec, clip_var = 10):
	return np.minimum(vec,clip_var)

def compute_truth(Data_cf,X,Y):
	groupped = np.array( Data_cf.groupby(X)[Y].mean().reset_index() )
	x_combination = groupped[:,:len(X)].astype(int)
	groupped_val = np.prod( np.array( Data_cf.groupby(X)[Y].mean().reset_index() )[:,len(X):], axis=1)
	result = np.column_stack((x_combination, groupped_val))
	return result

def performance_comparison(truth_mat, obs_mat, X):
	return np.mean( np.abs( truth_mat[:,len(X)] - obs_mat[:,len(X)] )  )

def DML_BD(Data, X, Y, G, L=5, seed=None, compare_with_others=False, clip_PW = False, add_noise=False, OM_misspecification=False, PW_misspecification=False):
	if seed is not None:
		random.seed(int(seed))
		np.random.seed(seed)

	param_X = {'objective': 'binary:logistic'}
	# param_Y = {'objective': 'reg:squarederror'}
	param_Y = {'objective': 'binary:logistic'}

	noise_vector = np.random.normal( len(Data) ** (-1/4) , len(Data) ** (-1/4), size = len(Data) )
	

	def train_PW(train_data, X, Z):
		model_PW = xgb.XGBRegressor(**param_X)
		model_PW.fit(train_data[Z], train_data[X])
		return model_PW

	def train_OM(train_data, X, Y, Z): 
		input_data = train_data.copy() if not OM_misspecification else pd.DataFrame(np.random.rand(*train_data.shape), columns=train_data.columns)
		model_OM = xgb.XGBRegressor(**param_Y)
		model_OM.fit(input_data[X + Z], input_data[Y])
		return model_OM

	def eval_OM(test_data, model_OM, X_ctf = None):
		# input_data = test_data.copy() if not OM_misspecification else pd.DataFrame(np.random.rand(*test_data.shape), columns=test_data.columns)
		input_data = test_data.copy()
		if X_ctf is not None: 
			input_data[X] = X_ctf
		result_OM = model_OM.predict(input_data[X + Z]) 
		return result_OM + noise_vector[:len(test_data)] if add_noise else result_OM

	def eval_PW(test_data, model_PW, x):
		input_data = test_data.copy() if not PW_misspecification else pd.DataFrame(np.random.rand(*test_data.shape), columns=test_data.columns)
		pred_X = model_PW.predict(input_data[Z])
		xVector = test_data[X].values.ravel() if x == 1 else 1-test_data[X].values.ravel()
		propensity_X = pred_X if x == 1 else 1-pred_X
		result_PW = (xVector / propensity_X)
		if clip_PW:
			result_PW = clip_vector( result_PW )
		return result_PW + noise_vector[:len(test_data)] if add_noise else result_PW

	Z = [col for col in Data.columns if col not in X and col not in Y]
	kf = KFold(n_splits=L)

	def compute_estimates(x):
		causal_mean_DML, causal_mean_PW, causal_mean_OM = 0, 0, 0
		for train_idx, test_idx in kf.split(Data):
			train_data, test_data = Data.iloc[train_idx], Data.iloc[test_idx]
		model_PW = train_PW(train_data, X, Z)
		model_OM = train_OM(train_data, X, Y, Z)

		pred_PW = eval_PW(test_data, model_PW, x)
		pred_OM = eval_OM(test_data, model_OM)
		pred_OM_x = eval_OM(test_data, model_OM, x)

		pseudo_output_DML = (pred_PW * (test_data[Y].values.ravel() - pred_OM)) + pred_OM_x

		# causal_mean_DML += value_in_min_max( np.mean(pseudo_output_DML) )
		# causal_mean_PW += value_in_min_max( np.mean( pred_PW * test_data[Y].values.ravel() ) )
		# causal_mean_OM += value_in_min_max( np.mean(pred_OM_x) )

		causal_mean_DML += np.mean(pseudo_output_DML) 
		causal_mean_PW += np.mean( pred_PW * test_data[Y].values.ravel() ) 
		causal_mean_OM += np.mean(pred_OM_x) 

		return value_in_min_max( causal_mean_DML / L ), value_in_min_max( causal_mean_PW / L ), value_in_min_max( causal_mean_OM / L )

	if not compare_with_others:
		return {x: compute_estimates(x)[0] for x in [0, 1]}
	else:
		return {
			'DML': {x: compute_estimates(x)[0] for x in [0, 1]},
			'PW': {x: compute_estimates(x)[1] for x in [0, 1]},
			'OM': {x: compute_estimates(x)[2] for x in [0, 1]}
		}

def DML_Napkin(Data, X, Y, G, L=5, seed=None, compare_with_others=False, clip_PW = False, add_noise=False, OM_misspecification=False, PW_misspecification=False):
	if seed is not None:
		random.seed(int(seed))
		np.random.seed(seed)

	Data_X = Data['X'].copy()
	r_value = (np.mean(Data['R']) > 0.5) * 1 

	def compute_estimates(x):
		Data['X'] = (x*Data_X + (1-x)*(1-Data_X)).copy()
		Data['XY'] = (Data['X'] * Data['Y']).copy()
		denominator = DML_BD(Data,['R'],['X'], L=L, seed=seed, compare_with_others=compare_with_others, clip_PW = clip_PW, add_noise=add_noise, OM_misspecification=OM_misspecification, PW_misspecification=PW_misspecification)
		numerator = DML_BD(Data,['R'],['XY'], L=L, seed=seed, compare_with_others=compare_with_others, clip_PW = clip_PW, add_noise=False, OM_misspecification=OM_misspecification, PW_misspecification=PW_misspecification)
		if not compare_with_others:
			return numerator[r_value] / denominator[r_value] 
		else:
			return  numerator['DML'][r_value] / denominator['DML'][r_value], numerator['PW'][r_value] / denominator['PW'][r_value], numerator['OM'][r_value] / denominator['OM'][r_value]

	# print(compute_estimates(0))
	
	if not compare_with_others:
		return {x: compute_estimates(x) for x in [0, 1]}
	else:
		return {
			'DML': {x: compute_estimates(x)[0] for x in [0, 1]},
			'PW': {x: compute_estimates(x)[1] for x in [0, 1]},
			'OM': {x: compute_estimates(x)[2] for x in [0, 1]}
		}

def DML_mSBD(Data, X, Y, G, L=5, seed=None, compare_with_others=False, clip_PW = False, add_noise=False, OM_misspecification=False, PW_misspecification=False):
	# Initialize the mSBD (modified Subset Back-door criterion) dictionaries and determine the number of treatment variables.
	dict_mSBD = mSBD.mSBD_result(G, X, Y)
	[dict_X, dict_Z, dict_Y] = dict_mSBD
	m = len(dict_X)  # Number of treatment variables.

	# Prepare for cross-validation
	kf = KFold(n_splits=L)
	x_combination = np.array(Data.groupby(X)[Y].mean().reset_index())[:, :len(X)].astype(int)

	# Set parameters for XGBoost models
	param_classification = {'objective': 'binary:logistic'}
	param_regression = {'objective': 'reg:squarederror'}

	noise_vector = np.random.normal( len(Data) ** (-1/4) , len(Data) ** (-1/4), size = len(Data) )

	# Set random seed if specified
	if seed is not None:
		random.seed(int(seed))
		np.random.seed(seed)

	# Helper functions
	def construct_Hi(dict_mSBD):
		[dict_X, dict_Z, dict_Y] = dict_mSBD
		dict_X["X0"] = set()
		dict_Z["Z0"] = set()
		dict_H = {"H0": dict_X["X0"] | dict_Y["Y0"] | dict_Z["Z0"]}
		
		m = len(dict_X)
		for j in range(1, m):
			dict_H[f"H{j}"] = dict_H[f"H{j-1}"] | dict_Z[f"Z{j}"]  | dict_X[f"X{j}"] | dict_Y[f"Y{j}"]  
		return dict_H

	dict_H = construct_Hi(dict_mSBD)

	def construct_I_y_Y(Data, dict_mSBD):
		Y_set = set()
		[dict_X, dict_Z, dict_Y] = dict_mSBD
		for i in range(m+1):
			Y_set |= dict_Y[f'Y{i}']
		Y_list = list(Y_set)
		IyY = 1
		for i, Yi in enumerate(Y_list):
			IyY *= (Data[Yi] == 1)
		return (IyY * 1).values.flatten()

	def train_OM_i(train_data, i, dict_H, eval_OM_i_plus_1_xfixed, param):
		[dict_X, dict_Z, dict_Y] = dict_mSBD
		H_i_minus_1 = list(dict_H[f"H{i-1}"])
		Xi = list(dict_X[f"X{i}"])
		Zi = list(dict_Z[f"Z{i}"])
		input_data = train_data.copy() if not OM_misspecification else pd.DataFrame(np.random.rand(*train_data.shape), columns=train_data.columns)
		input_for_OM_i = input_data[H_i_minus_1 + Xi + Zi]
		if np.all(np.logical_or(eval_OM_i_plus_1_xfixed == 0, eval_OM_i_plus_1_xfixed == 1)):
			param = param_classification
		else:
			param = param_regression
		model_OM_i = xgb.XGBRegressor(**param)
		model_OM_i.fit(input_for_OM_i, eval_OM_i_plus_1_xfixed)
		return model_OM_i

		# (train_data, i, dict_H, model_OM[i], x[i], dict_X)
	def eval_OM_i(data, i, dict_H, model_OM_i, xi, my_noise):
		[dict_X, dict_Z, dict_Y] = dict_mSBD
		H_i_minus_1 = list(dict_H[f"H{i-1}"])
		Xi = list(dict_X[f"X{i}"])
		Zi = list(dict_Z[f"Z{i}"])
		input_for_OM_i = data[H_i_minus_1 + Xi + Zi].copy()
		if xi is not None:
			input_for_OM_i[Xi] = xi
		result_OM_i = model_OM_i.predict(input_for_OM_i)
		return result_OM_i + noise_vector[:len(data)] if train_noise else result_OM_i

	def train_PW_i(train_data, i, dict_H, param_classification):
		[dict_X, dict_Z, dict_Y] = dict_mSBD
		H_i_minus_1 = list(dict_H[f"H{i-1}"])
		Xi = list(dict_X[f"X{i}"])
		Zi = list(dict_Z[f"Z{i}"])
		input_for_PW_i = train_data[H_i_minus_1 + Zi]
		output_for_PW_i = train_data[Xi]
		model_PW_i = xgb.XGBRegressor(**param_classification)
		model_PW_i.fit(input_for_PW_i, output_for_PW_i)
		return model_PW_i

	def eval_PW_i(test_data, i, dict_H, model_PW_i, xi):
		[dict_X, dict_Z, dict_Y] = dict_mSBD
		H_i_minus_1 = list(dict_H[f"H{i-1}"])
		Xi = list(dict_X[f"X{i}"])
		Zi = list(dict_Z[f"Z{i}"])
		input_data = test_data.copy() if not PW_misspecification else pd.DataFrame(np.random.rand(*test_data.shape), columns=test_data.columns)
		input_for_PW_i = input_data[H_i_minus_1 + Zi]
		pred_PW_i = model_PW_i.predict(input_for_PW_i)
		propensity_PW_i = xi * pred_PW_i + (1 - xi) * pred_PW_i
		Xi_data = (xi * test_data[Xi] + (1 - xi) * (1 - test_data[Xi])).values.flatten()
		result_PW_i = Xi_data / propensity_PW_i
		if clip_PW:
			result_PW_i = clip_vector( result_PW_i )
		return result_PW_i + noise_vector[:len(test_data)] if add_noise else result_PW_i

	def compute_DML(model_OM, model_PW, test_data, x, dict_H):
		eval_PW_product = {}
		eval_OM = {}
		eval_OM_x = {m+1: construct_I_y_Y(test_data, dict_mSBD)}

		# OM and PW evaluations
		for i in range(m, 0, -1):
			xi = x[i-1]
			eval_OM[i] = eval_OM_i(test_data, i, dict_H, model_OM[i], None, train_noise = add_noise)
			eval_OM_x[i] = eval_OM_i(test_data, i, dict_H, model_OM[i], xi, train_noise = add_noise)

		eval_PW_product[1] = eval_PW_i(test_data, 1, dict_H, model_PW[1], x[0])
		for i in range(2, m+1):
			xi = x[i-1]
			eval_PW_product[i] = eval_PW_product[i-1] * eval_PW_i(test_data, i, dict_H, model_PW[i], xi)

		# Compute pseudo-outcome
		pseudo_outcome_DML = np.zeros(len(test_data))
		for i in range(m, 0, -1):
			pseudo_outcome_DML += (eval_PW_product[i] * (eval_OM_x[i+1] - eval_OM[i]))
		
		pseudo_outcome_DML += eval_OM_x[1]
		pseudo_outcome_OM = eval_OM_x[1]
		pseudo_outcome_PW = eval_PW_product[m] * eval_OM_x[m+1]

		return np.mean(pseudo_outcome_DML), np.mean(pseudo_outcome_PW), np.mean(pseudo_outcome_OM)

	def compute_estimates(x):
		causal_mean_DML, causal_mean_PW, causal_mean_OM = 0, 0, 0
		DML_value_test = 0
		for train_idx, test_idx in kf.split(Data):
			train_data, test_data = Data.iloc[train_idx], Data.iloc[test_idx]
			IyY = construct_I_y_Y(train_data, dict_mSBD) 
			dict_H = construct_Hi(dict_mSBD)

			eval_OM_x = {m+1: IyY}
			model_OM = {}
			model_PW = {}
			
			# Train and evaluate OM models
			for i in range(m, 0, -1):
				xi = x[i-1]
				model_OM[i] = train_OM_i(train_data, i, dict_H, eval_OM_x[i+1], param_regression)
				eval_OM_x[i] = eval_OM_i(train_data, i, dict_H, model_OM[i], xi, train_noise = False)

			# Train PW models
			for i in range(1, m+1):
				model_PW[i] = train_PW_i(train_data, i, dict_H, param_classification)
			
			# Compute DML value for test data
			DML_estimate, PW_estimate, OM_estimate = compute_DML(model_OM, model_PW, test_data, x, dict_H)
			causal_mean_DML += DML_estimate
			causal_mean_PW += PW_estimate
			causal_mean_OM += OM_estimate

		return value_in_min_max( causal_mean_DML / L ), value_in_min_max( causal_mean_PW / L ), value_in_min_max( causal_mean_OM / L )

	if not compare_with_others:
		result_x = []
		for x in x_combination:
			result_x.append( compute_estimates(x) )
		result = np.column_stack((x_combination, result_x))
		return result
	else:
		result_DML = []
		result_PW = []
		result_OM = []
		for x in x_combination:
			result_x = compute_estimates(x)
			result_DML.append( result_x[0] )
			result_PW.append( result_x[1] )
			result_OM.append( result_x[2] )
		result_DML = np.column_stack((x_combination, result_DML))
		result_PW = np.column_stack((x_combination, result_PW))
		result_OM = np.column_stack((x_combination, result_OM))
		return {'DML': result_DML, 'PW': result_PW, 'OM': result_OM}



if __name__ == "__main__":
	# None
	X = ['X1','X2']
	Y = ['Y1','Y2']

	fixed_simulation_seed = 240209 # given number
	random.seed(fixed_simulation_seed)
	np.random.seed(fixed_simulation_seed)

	num_samples = 50000

	scenario = 3
	if scenario == 1: 
		add_noise = False
		OM_misspecification = False
		PW_misspecification = False
	elif scenario == 2:
		add_noise = True
		OM_misspecification = False
		PW_misspecification = False
	elif scenario == 3:
		add_noise = False
		OM_misspecification = True
		PW_misspecification = False
	elif scenario == 4:
		add_noise = False
		OM_misspecification = False
		PW_misspecification = True

	[scm, G, Data, Data_cf] = example_SCM.mSBD_SCM(num_samples, generate_submodel = True, seed = fixed_simulation_seed)
	tmp = DML_mSBD(Data, X, Y, G,  L=2, seed=fixed_simulation_seed, compare_with_others=True, clip_PW = False, add_noise=add_noise, OM_misspecification=OM_misspecification, PW_misspecification=PW_misspecification)
	truth_mat = compute_truth(Data_cf,X,Y)

	for (idx, key) in enumerate(tmp):
		print( key,  performance_comparison(truth_mat, tmp[key], X) )

	# Data['Y1Y2'] = Data['Y1'] * Data['Y2']
	# Data_cf['Y1Y2'] = Data_cf['Y1'] * Data_cf['Y2']
	# Y = ['Y1Y2']

	

	# tmp = DML_Napkin(Data,X,Y)
	# tmp2 = DML_Napkin(Data,X,Y,compare_with_others=True)
	# tmp3 = DML_Napkin(Data,X,Y,compare_with_others=True,clip_PW=True)

	# naive = DML_BD(Data,X,Y,compare_with_others=True)

	# # Data['X'] = (1-Data['X']).copy()
	# Data['XY'] = Data['X'] * Data['Y']

	# tmp = DML_BD(Data,['R'],['X'],compare_with_others=True)
	# tmp2 = DML_BD(Data,['R'],['XY'],compare_with_others=True)


	# print(tmp2[0]/tmp[0], tmp2[1]/tmp[0])

	# tmp2['DML'][0]/tmp['DML'][0]
	# tmp2['PW'][0]/tmp['PW'][0]
	# tmp2['OM'][0]/tmp['OM'][0]
	# tmp2['DML'][1]/tmp['DML'][1]
	# tmp2['PW'][1]/tmp['PW'][1]
	# tmp2['OM'][1]/tmp['OM'][1]


	# X = ['X']
	# Y = ['Y']

	# num_simulation = 10 # given number
	# list_DGP_seeds = list(np.random.randint(1,100000,size=num_simulation))

	# list_num_samples = [500, 10000, 25000, 50000]

	# [_, _, _, Data_cf] = example_SCM.BD_SCM(list_num_samples[0], generate_submodel = True, seed = fixed_simulation_seed)

	# dict_truth = compute_truth(Data_cf,X,Y)

	# performance_dict_DML = {num_sample: [] for num_sample in list_num_samples}
	# performance_dict_PW = {num_sample: [] for num_sample in list_num_samples}
	# performance_dict_OM = {num_sample: [] for num_sample in list_num_samples}

	# performance_dict = {}
	# performance_dict['DML'] = performance_dict_DML
	# performance_dict['PW'] = performance_dict_PW
	# performance_dict['OM'] = performance_dict_OM

	# compare_with_others = True
	# clip_PW = False

	# scenario = 1
	# if scenario == 1: 
	# 	add_noise = False
	# 	OM_misspecification = False
	# 	PW_misspecification = False
	# elif scenario == 2:
	# 	add_noise = True
	# 	OM_misspecification = False
	# 	PW_misspecification = False
	# elif scenario == 3:
	# 	add_noise = False
	# 	OM_misspecification = True
	# 	PW_misspecification = False
	# elif scenario == 4:
	# 	add_noise = False
	# 	OM_misspecification = False
	# 	PW_misspecification = True

	# print(f"Run simulation with scenario {scenario}" )
	# num_sample = 1000
	# [scm, G, Data, _] = example_SCM.BD_SCM(num_sample, generate_submodel = False, seed = 123)
	# dict_obs = DML_BD(Data, X, Y, L = 2, seed = 123, compare_with_others = compare_with_others, clip_PW = clip_PW, add_noise = add_noise, OM_misspecification = OM_misspecification, PW_misspecification = PW_misspecification)

	# for each_seed in tqdm(list_DGP_seeds, desc = "Simulating seeds"):	
	# 	for num_sample in list_num_samples:
	# 		[scm, G, Data, _] = example_SCM.BD_SCM(num_sample, generate_submodel = False, seed = each_seed)
	# 		dict_obs = DML_BD(Data, X, Y, L = 2, seed = each_seed, compare_with_others = compare_with_others, clip_PW = clip_PW,
	# 			add_noise = add_noise, OM_misspecification = OM_misspecification, PW_misspecification = PW_misspecification)

	# 		for estimator in dict_obs.keys():
	# 			performance_dict[estimator][num_sample].append( performance_comparison(dict_truth, dict_obs[estimator]) )
			
	

	# def plot_convergence(performance_dict, list_num_samples):
	# 	plt.figure(figsize=(8, 12))
	# 	# Separate data for each estimator
	# 	performance_dml = performance_dict['DML']
	# 	performance_pw = performance_dict['PW']
	# 	performance_om = performance_dict['OM']
	# 	# Compute average performance for each sample size
	# 	avg_performance_dml = [np.mean(performance_dml[n]) for n in list_num_samples]
	# 	avg_performance_pw = [np.mean(performance_pw[n]) for n in list_num_samples]
	# 	avg_performance_om = [np.mean(performance_om[n]) for n in list_num_samples]
	# 	std_error_dml = [np.std(performance_dml[n])/np.sqrt(len(performance_dml[n])) for n in list_num_samples]
	# 	std_error_pw = [np.std(performance_pw[n])/np.sqrt(len(performance_pw[n])) for n in list_num_samples]
	# 	std_error_om = [np.std(performance_om[n])/np.sqrt(len(performance_om[n])) for n in list_num_samples]
	# 	# Plot each estimator
	# 	# Plot each estimator with error bars
	# 	plt.errorbar(list_num_samples, avg_performance_dml, yerr=std_error_dml, label='DML', color='blue', marker='o', capsize=5)
	# 	plt.errorbar(list_num_samples, avg_performance_pw, yerr=std_error_pw, label='PW', color='green', marker='s', capsize=5)
	# 	plt.errorbar(list_num_samples, avg_performance_om, yerr=std_error_om, label='OM', color='red', marker='^', capsize=5)
	# 	# plt.plot(list_num_samples, avg_performance_dml, label='DML', color='blue', marker='o')
	# 	# plt.plot(list_num_samples, avg_performance_pw, label='PW', color='green', marker='s')
	# 	# plt.plot(list_num_samples, avg_performance_om, label='OM', color='red', marker='^')
	# 	# plt.xlabel('Number of Samples')
	# 	# plt.ylabel('Error')
	# 	custom_xticks = list_num_samples.copy()
	# 	custom_yticks = [0.05, 0.1, 0.15, 0.2]
	# 	plt.xticks(custom_xticks,fontsize=20)  # Adjust the fontsize as needed for x-axis tick labels
	# 	plt.yticks(custom_yticks, fontsize=20)  # Adjust the fontsize as needed for y-axis tick labels
	# 	# plt.title('Convergence Comparison of DML, PW, OM Estimators')
	# 	# plt.legend()
	# 	plt.grid(False)
	# 	plt.show()

	# plot_convergence(performance_dict, list_num_samples)






