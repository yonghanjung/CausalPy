import networkx as nx
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import random
import matplotlib.pyplot as plt
import pickle
from contextlib import contextmanager

import graph
import adjustment    
import example_SCM
import est_mSBD
import statmodules
# from est_mSBD import xgb_predict
# from statmodules import entropy_balancing_osqp
import est_general 


# Context manager to simulate scenarios
@contextmanager
def simulate_scenario(scenario):
	original_xgb_predict = est_mSBD.xgb_predict
	original_entropy_balancing_osqp = statmodules.entropy_balancing_osqp
	
	if scenario == 2: 
		# Modify the train_ML_model to add noise
		def noisy_predict(model, data, col_feature):
			pred = original_xgb_predict(model, data, col_feature)
			noise_mean = data.shape[0] ** (-1/4)
			noise_scale = data.shape[0] ** (-1/4)
			return pred + np.random.normal(loc = noise_mean, scale = noise_scale, size = len(data))

		def noisy_EB(obs, x_val, X, Z, col_feature_1, col_feature_2):
			pred = original_entropy_balancing_osqp(obs, x_val, X, Z, col_feature_1, col_feature_2)
			noise_mean = obs.shape[0] ** (-1/4)
			noise_scale = obs.shape[0] ** (-1/4)
			return pred + np.random.normal(loc = noise_mean, scale = noise_scale, size = len(obs))

		est_mSBD.xgb_predict = noisy_predict
		statmodules.entropy_balancing_osqp = noisy_EB

		yield 

		est_mSBD.xgb_predict = original_xgb_predict
		statmodules.entropy_balancing_osqp = original_entropy_balancing_osqp


	elif scenario == 3: 
		def contimated_predict(model, data, col_feature):
			random_data = pd.DataFrame(np.random.rand(data.shape[0], data.shape[1]), columns=data.columns)
			return original_xgb_predict(model, random_data, col_feature)

		est_mSBD.xgb_predict = contimated_predict

		yield 

		est_mSBD.xgb_predict = original_xgb_predict

	elif scenario == 4: 
		def contimated_balancing(obs, x_val, X, Z, col_feature_1, col_feature_2): 
			random_data = np.random.rand(len(obs))
			return np.clip(random_data, a_min=0, a_max = None)

		statmodules.entropy_balancing_osqp = contimated_balㅇncing

		yield 

		statmodules.entropy_balancing_osqp = original_entropy_balancing_osqp

def run_DML_simulation(simulation_round, list_num_samples, scenario, seednum, pkl_path, scm, X, Y, filename):
	random.seed(seednum)
	np.random.seed(seednum)

	list_seeds = list(np.random.randint(1,100000,size=simulation_round))
	
	G = scm.graph
	G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
	topo_V = graph.find_topological_order(G)

	y_val = np.ones(len(Y)).astype(int)
	truth = statmodules.ground_truth(scm, X, Y, y_val)

	obs_data = scm.generate_samples(1000, seed=seednum)[topo_V]

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
	with simulate_scenario(4):
		ATE = est_general.estimate_case_by_case(G, X, Y, y_val, obs_data)
		performance_table, rank_correlation_table = statmodules.compute_performance(truth, ATE)
		
		print(f"Performance")
		print(performance_table)
		print(f"Rank Correlation")
		print(rank_correlation_table)



