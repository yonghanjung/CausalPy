import networkx as nx
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import graph
import adjustment    
import example_SCM
import estimation

def compute_truth(Data_cf,X,Y):
	groupped = np.array( Data_cf.groupby(X)[Y].mean().reset_index() )
	x_combination = groupped[:,:len(X)].astype(int)
	groupped_val = np.prod( np.array( Data_cf.groupby(X)[Y].mean().reset_index() )[:,len(X):], axis=1)
	result = np.column_stack((x_combination, groupped_val))
	return result

def performance_comparison(dict_truth, dict_obs):
	performance = 0
	for key in dict_obs.keys():
		performance += np.abs(dict_truth[key] - dict_obs[key])
	return performance / len(dict_obs.keys())

def performance_comparison(truth_mat, obs_mat, X):
	return np.mean( np.abs( truth_mat[:,len(X)] - obs_mat[:,len(X)] )  )

def DML_scenario(scenario):
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
	return add_noise, OM_misspecification, PW_misspecification

def individual_simulation():
	# sim_DGP = example_SCM.BD_SCM
	# sim_instance = estimation.DML_BD

	# sim_DGP = example_SCM.Napkin_SCM
	# sim_instance = estimation.DML_Napkin

	sim_DGP = example_SCM.mSBD_SCM
	sim_instance = estimation.DML_mSBD
	return sim_DGP,sim_instance

def run_DML_simulation(num_simulation, list_num_samples, scenario, fixed_seed, pkl_path, *args):
	''' parameters in *args
	X = X, Y = Y, L=L, clip_PW = False, filename 
	'''
	if args: 
		[X, Y, L, clip_PW, filename] = args[0]

	add_noise, OM_misspecification, PW_misspecification = DML_scenario(scenario)
	sim_DGP, sim_instance = individual_simulation()

	random.seed(fixed_seed)
	np.random.seed(fixed_seed)

	list_seeds = list(np.random.randint(1,100000,size=num_simulation))
	[_, G, _, Data_cf] = sim_DGP(100, generate_submodel = True, seed = fixed_seed)

	truth_mat = compute_truth(Data_cf,X,Y)

	performance_dict_DML = {num_sample: [] for num_sample in list_num_samples}
	performance_dict_PW = {num_sample: [] for num_sample in list_num_samples}
	performance_dict_OM = {num_sample: [] for num_sample in list_num_samples}

	performance_dict = {}
	performance_dict['DML'] = performance_dict_DML
	performance_dict['PW'] = performance_dict_PW
	performance_dict['OM'] = performance_dict_OM

	for each_seed in tqdm(list_seeds, desc = "Simulating seeds"):	
		for num_sample in list_num_samples:
			[scm, G, Data, _] = sim_DGP(num_sample, generate_submodel = False, seed = each_seed)
			dict_obs = sim_instance(Data, X, Y, G, L = L, seed = each_seed, compare_with_others = True, clip_PW = clip_PW,
				add_noise = add_noise, OM_misspecification = OM_misspecification, PW_misspecification = PW_misspecification)

			for estimator in dict_obs.keys():
				performance_dict[estimator][num_sample].append( performance_comparison(truth_mat, dict_obs[estimator], X) )

	if pkl_path is not None:
		result_file_name = pkl_path + "result_" + filename + ".pkl"
		with open(result_file_name, 'wb') as file:
			pickle.dump(performance_dict, file)
		
		param_file_name = pkl_path + "parameters_" + filename + ".pkl"
		parameters = {"num_simulation": num_simulation, "list_num_samples": list_num_samples, "scenario": scenario, "fixed_seed": fixed_seed, "X": X, "Y": Y, "L": L, "clip_PW": clip_PW}
		with open(param_file_name, 'wb') as paramfile:
			pickle.dump(parameters, paramfile)
	
	return performance_dict

def loaded_result(pkl_path, filename):
	filename = pkl_path + filename
	with open(filename, 'rb') as file: 
		loaded_data = pickle.load(file)
		return loaded_data

def draw_plots(performance_dict, fig_path = None, *args):
	''' args example
	figsize = (8,12)
	custom_xticks = True or False 
	custom_yticks = [0.05, 0.1, 0.15, 0.2] or None
	'''
	if args: 
		[figsize, custom_xticks, custom_yticks, ylim, filename] = args[0]

	list_num_samples = list(performance_dict[list(performance_dict.keys())[0]].keys())
	plt.figure(figsize=figsize)

	# Separate data for each estimator
	performance_dml = performance_dict['DML']
	performance_pw = performance_dict['PW']
	performance_om = performance_dict['OM']

	# Compute average performance for each sample size
	avg_performance_dml = [np.mean(performance_dml[n]) for n in list_num_samples]
	avg_performance_pw = [np.mean(performance_pw[n]) for n in list_num_samples]
	avg_performance_om = [np.mean(performance_om[n]) for n in list_num_samples]
	std_error_dml = [np.std(performance_dml[n])/np.sqrt(len(performance_dml[n])) for n in list_num_samples]
	std_error_pw = [np.std(performance_pw[n])/np.sqrt(len(performance_pw[n])) for n in list_num_samples]
	std_error_om = [np.std(performance_om[n])/np.sqrt(len(performance_om[n])) for n in list_num_samples]
	
	# Plot each estimator with error bars
	plt.errorbar(list_num_samples, avg_performance_dml, yerr=std_error_dml, label='DML', color='blue', marker='o', capsize=5)
	plt.errorbar(list_num_samples, avg_performance_pw, yerr=std_error_pw, label='PW', color='green', marker='s', capsize=5)
	plt.errorbar(list_num_samples, avg_performance_om, yerr=std_error_om, label='OM', color='red', marker='^', capsize=5)

	# plt.plot(list_num_samples, avg_performance_dml, label='DML', color='blue', marker='o')
	# plt.plot(list_num_samples, avg_performance_pw, label='PW', color='green', marker='s')
	# plt.plot(list_num_samples, avg_performance_om, label='OM', color='red', marker='^')
	# plt.xlabel('Number of Samples')
	# plt.ylabel('Error')

	# custom_xticks = list_num_samples.copy()
	# custom_yticks = [0.05, 0.1, 0.15, 0.2]
	if custom_xticks is True: 
		plt.xticks(list_num_samples,fontsize=20)  # Adjust the fontsize as needed for x-axis tick labels
	if custom_yticks is not None:
		plt.yticks(custom_yticks, fontsize=20)  # Adjust the fontsize as needed for y-axis tick labels
	# plt.title('Convergence Comparison of DML, PW, OM Estimators')
	# plt.legend()
	if ylim is not None:
		plt.ylim(ylim)
	plt.grid(False)

	if fig_path is not None:
		fig_file_name = fig_path + "plot_" + filename + ".png"
		plt.savefig(fig_file_name)

	plt.show()

if __name__ == "__main__":
	num_simulation = 20
	list_num_samples = [500, 10000, 25000, 50000]
	scenario = 4
	fixed_seed = 240212
	 
	X = ['X1','X2']
	Y = ['Y1','Y2']
	L = 2 
	clip_PW = False
	pkl_path = "pkl/"
	filename = f"240212-1415-scenario-{scenario}-DMLmSBD"

	other_params = [X, Y, L, clip_PW, filename]
	# X = X, Y = Y, L=L, clip_PW = False, filename 
	performance_dict = run_DML_simulation(num_simulation, list_num_samples, scenario, fixed_seed, pkl_path, other_params)

	# [figsize, custom_xticks, custom_yticks, filename]
	fig_path = "plot/"
	fig_params = [(8,10), True, [0.05, 0.15, 0.20, 0.25], (0,0.25), filename]
	draw_plots(performance_dict, fig_path, fig_params)	

	# loading
	# scenario = 4
	# filename = f"240209-2100-scenario-{scenario}-DMLNapkin"
	# performance_dict = loaded_result("pkl/", "result_" + filename + ".pkl")
	# fig_path = "plot/"
	# fig_params = [(8,8), True, [0.025, 0.05, 0.075, 0.1], (0,0.1), filename]
	# draw_plots(performance_dict, fig_path, fig_params)


