import networkx as nx
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import random
import matplotlib.pyplot as plt
import dill as pickle
from contextlib import contextmanager
from tqdm import tqdm
import sys

import graph
import adjustment    
import example_SCM
import est_mSBD
import statmodules
import est_general
import identify 

import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)


# Context manager to simulate scenarios
@contextmanager
def simulate_scenario(scenario):
	original_xgb_predict = est_mSBD.xgb_predict
	original_entropy_balancing_osqp = statmodules.entropy_balancing_osqp

	if scenario == 1:
		# Do nothing
		yield 

		est_mSBD.xgb_predict = original_xgb_predict
		statmodules.entropy_balancing_osqp = original_entropy_balancing_osqp
	
	if scenario == 2: 
		# Modify the train_ML_model to add noise
		def noisy_predict(model, data, col_feature):
			pred = original_xgb_predict(model, data, col_feature)
			noise_mean = data.shape[0] ** (-1/4)
			noise_scale = data.shape[0] ** (-1/4)
			return pred + np.random.normal(loc = noise_mean, scale = noise_scale, size = len(data))

		def noisy_EB(obs, x_val, X, Z, col_feature_1, col_feature_2):
			pred = original_entropy_balancing_osqp(obs, x_val, X, Z, col_feature_1, col_feature_2)
			if pred is None:
				# Handle the case where pred is None
				raise ValueError("The original entropy balancing function returned None")
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

		def contimated_predict2(model, data, col_feature):
			random_data = pd.DataFrame(np.random.rand(data.shape[0], data.shape[1]), columns=data.columns)
			orig_predict = original_xgb_predict(model, random_data, col_feature)
			return np.zeros(len(orig_predict))

		# est_mSBD.xgb_predict = contimated_predict
		est_mSBD.xgb_predict = contimated_predict2

		yield 

		est_mSBD.xgb_predict = original_xgb_predict

	elif scenario == 4: 
		def contimated_balancing(obs, x_val, X, Z, col_feature_1, col_feature_2): 
			random_data = np.random.rand(len(obs))
			return np.clip(random_data, a_min=0, a_max = None)

		def contimated_balancing2(obs, x_val, X, Z, col_feature_1, col_feature_2): 
			random_data = np.zeros(len(obs))
			return np.clip(random_data, a_min=0, a_max = None)

		# statmodules.entropy_balancing_osqp = contimated_balancing
		statmodules.entropy_balancing_osqp = contimated_balancing2

		yield 

		statmodules.entropy_balancing_osqp = original_entropy_balancing_osqp

def run_DML_simulation(simulation_round, list_num_samples, list_of_estimators, scenario, seednum, scm, X, Y, pkl_path, filename):
	random.seed(seednum)
	np.random.seed(seednum)

	print(f'Simulation on Scenario {scenario} with a Base seednum {seednum} with Filename: {filename}')

	list_seeds = list(np.random.randint(1,100000,size=simulation_round))
	
	G = scm.graph
	G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
	topo_V = graph.find_topological_order(G)

	y_val = np.ones(len(Y)).astype(int)
	truth = statmodules.ground_truth(scm, X, Y, y_val)

	performance_dict = {}

	for estimator in list_of_estimators: 
		performance_dict[estimator] = {}
		for num_sample in list_num_samples:
			performance_dict[estimator][num_sample] = []

	for each_seed in tqdm(list_seeds, desc = "Simulating seeds"):	
		for num_sample in list_num_samples:
			obs_data = scm.generate_samples(num_sample, seed=each_seed)[topo_V]
			with simulate_scenario(scenario):
				ATE = est_general.estimate_case_by_case(G, X, Y, y_val, obs_data)
				_, _, performance_dict_per_seed, _ = statmodules.compute_performance(truth, ATE)

				for estimator in list_of_estimators: 
					performance_dict[estimator][num_sample].append( performance_dict_per_seed[estimator] )
	
	if pkl_path is not None:
		result_file_name = pkl_path + "result_"  + filename + ".pkl"
		with open(result_file_name, 'wb') as file:
			pickle.dump(performance_dict, file)
		
		param_file_name = pkl_path + "parameters_" + filename + ".pkl"
		parameters = {"simulation_round": simulation_round, "list_num_samples": list_num_samples, "list_of_estimators": list_of_estimators, "scenario": scenario, 
						"seednum": seednum, "scm": scm, "X": X, "Y": Y, "pkl_path": pkl_path, "filename": filename}
		with open(param_file_name, 'wb') as paramfile:
			pickle.dump(parameters, paramfile)

	return performance_dict


def loaded_result(pkl_path, filename):
	filename = pkl_path + filename
	with open(filename, 'rb') as file: 
		loaded_data = pickle.load(file)
		return loaded_data

def draw_plots(performance_dict, **kwargs):
	''' 
	kwargs example
	fig_size = (8,12)
	custom_xticks = True or False 
	custom_yticks = [0.05, 0.1, 0.15, 0.2] or None
	list_num_samples = List of sample sizes
	list_of_estimators = List of estimators to plot
	'''
	# Extract arguments from kwargs
	list_num_samples = kwargs.pop('list_num_samples', None)
	if list_num_samples is None: 
		list_num_samples = list(performance_dict[list(performance_dict.keys())[0]].keys())

	list_of_estimators = kwargs.pop('list_of_estimators', None)
	if list_of_estimators is None:
		list_of_estimators = list(performance_dict.keys())

	fig_size = kwargs.pop('fig_size', (8, 12))
	fig_filename = kwargs.pop('fig_filename', None)
	fig_path = kwargs.pop('fig_path', None)
	fontsize_xtick = kwargs.pop('fontsize_xtick', None)
	fontsize_ytick = kwargs.pop('fontsize_ytick', None)
	legend_on = kwargs.pop('legend_on', None)

	# Create figure
	plt.figure(figsize=fig_size)

	# Colors and markers for different estimators
	colors = ['blue', 'green', 'red', 'purple', 'orange']
	markers = ['o', 's', '^', 'D', 'v']

	# Plot each estimator's performance dynamically
	for idx, estimator in enumerate(list_of_estimators):
		performance = performance_dict[estimator]
		avg_performance = [np.mean(performance[n]) for n in list_num_samples]
		std_error = np.clip([np.std(performance[n])/np.sqrt(len(performance[n])) for n in list_num_samples], 0, 1)

		# Use a different color and marker for each estimator
		plt.errorbar(list_num_samples, avg_performance, yerr=std_error, label=estimator, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], capsize=5, linewidth=3)

	# Add labels, legend, and other plot customizations
	plt.xlabel('')
	plt.ylabel('')
	if legend_on:
		plt.legend(fontsize=fontsize_xtick)
	else:
		plt.legend([])

	plt.grid(False)
	plt.xticks(list_num_samples, fontsize=fontsize_xtick)
	plt.yticks(fontsize=fontsize_ytick)
	
	# Save figure if a path and filename are provided
	if fig_path is not None and fig_filename is not None:
		plt.savefig(fig_path + fig_filename)

	plt.show()


def call_examples(example_number):

	if example_number == 1:
		# Back-door 
		scm, X, Y = example_SCM.BD_SCM(seednum = seednum)	
		example_name = 'CanonBD'

	elif example_number == 2:
		# mSBD
		scm, X, Y = example_SCM.mSBD_SCM(seednum = seednum)	
		example_name = 'CanonMSBD'

	elif example_number == 3:
		# Front-door
		scm, X, Y = example_SCM.FD_SCM(seednum = seednum)	
		example_name = 'CanonFD'

	elif example_number == 4: 
		# PlanID
		scm, X, Y = example_SCM.Plan_ID_SCM(seednum = seednum)
		example_name = 'CanonPlanID'

	elif example_number == 5: 
		# Napkin
		scm, X, Y = example_SCM.Napkin_SCM(seednum = seednum)
		example_name = 'CanonNapkin'

	elif example_number == 6: 
		# Napkin_FD_SCM
		scm, X, Y = example_SCM.Napkin_FD_SCM(seednum = seednum)
		example_name = 'CanonNapkinFD'

	elif example_number == 7: 
		# Nested Napkin
		scm, X, Y = example_SCM.Nested_Napkin_SCM(seednum = seednum)
		example_name = 'CanonNestedNapkinFD'

	elif example_number == 8: 
		# Nested Napkin
		scm, X, Y = example_SCM.Double_Napkin_SCM(seednum = seednum)
		example_name = 'CanonDoubleNapkin'

	# elif example_number == 9: 
	# 	# Nested Napkin
	# 	scm, X, Y = example_SCM.Double_Napkin_SCM(seednum = seednum)
	# 	example_name = 'CanonDoubleNapkin'

	return scm, X, Y, example_name



if __name__ == "__main__":
	''' 
	===== Back-door =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 1 240911 0000 
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 1 240911 0000 
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 1 240911 0000 
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 1 240911 0000 

	''' 
	===== mSBD =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 2 240911 0000 
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 2 240911 0000 
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 2 240911 0000 
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 2 240911 0000 

	''' 
	===== FD =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 3 240911 0000 
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 3 240911 0000 
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 3 240911 0000 
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 3 240911 0000 

	''' 
	===== PlanID =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 4 240911 0000 
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 4 240911 0000 
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 4 240911 0000 
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 4 240911 0000 

	''' 
	===== Napkin =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 5 240916 0000 
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 5 240916 0000 
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 5 240916 0000 
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 5 240916 0000 

	''' 
	===== Napkin FD =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 6 240911 0000 
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 6 240911 0000 
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 6 240911 0000 
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 6 240911 0000 

	''' 
	===== Nested Napkin =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 7 240916 0000 
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 7 240916 0000 
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 7 240916 0000 
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 7 240916 0000 

	''' 
	===== Double Napkin =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 8 240917 0000 
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 8 240917 0000 
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 8 240917 0000 
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 8 240917 0000 

	seednum = int(sys.argv[1])

	np.random.seed(seednum)
	random.seed(seednum)

	simulation_round = int(sys.argv[2])
	list_num_samples = [100, 20000, 50000, 100000]
	# list_num_samples = [100, 1000, 10000]
	list_of_estimators = ['OM', 'IPW', 'DML']
	scenario = int(sys.argv[3])

	example_number = int(sys.argv[4])
	scm, X, Y, example_name = call_examples(example_number)
	
	sim_date = int(sys.argv[5])
	sim_time = int(sys.argv[6])

	pkl_path = 'log_experiments/pkl/'
	fig_path = 'log_experiments/plot/'

	filename = f'{sim_date}{sim_time}_{example_name}_seednum{seednum}_scenario{scenario}_round{simulation_round}'

	print(f'base_seed: {seednum}, simulation round: {simulation_round}, scenario: {scenario}, example_number: {example_number}, example_name: {example_name}, sim_date_time: {sim_date}_{sim_time}')

	pkl_extension = '.pkl'
	fig_extension = '.png'

	pkl_filename = filename + pkl_extension
	fig_filename = filename + fig_extension
	fig_size = (12,8)

	performance_dict = run_DML_simulation(simulation_round, list_num_samples, list_of_estimators, scenario, seednum, scm, X, Y, pkl_path, pkl_filename)

	if scenario == 1:
		legend_on = True
	else:
		legend_on = False

	draw_plots(performance_dict, fig_size = fig_size, fig_path = fig_path, fig_filename = fig_filename, fontsize_xtick = 30, fontsize_ytick = 30, legend_on = legend_on)

