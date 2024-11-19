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
import random_generator

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
	
	elif scenario == 2: 
		# Modify the train_ML_model to add noise
		def noisy_predict(model, data, col_feature):
			pred = original_xgb_predict(model, data, col_feature)
			noise_mean = data.shape[0] ** (-1/4)
			noise_scale = data.shape[0] ** (-1/4)
			scale_param = np.max([1.0, np.mean(pred)])
			return pred + scale_param * np.random.normal(loc = noise_mean, scale = noise_scale, size = len(data))

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
			return np.ones(len(data))
			# return np.clip(random_data, a_min=0, a_max = None)
			# random_data = pd.DataFrame(np.random.rand(data.shape[0], data.shape[1]), columns=data.columns)
			# orig_predict = original_xgb_predict(model, random_data, col_feature)
			# return np.zeros(len(orig_predict))

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

def run_DML_simulation(simulation_round, list_num_samples, list_of_estimators, scenario, seednum, scm, X, Y, pkl_path, filename, **kwargs):
	random.seed(seednum)
	np.random.seed(seednum)

	cluster_variables = kwargs.get('cluster_variables', None)

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
			df_SCM = scm.generate_samples(num_sample, seed=each_seed)
			observables = [node for node in df_SCM.columns if not node.startswith('U')]
			obs_data = df_SCM[observables]
			with simulate_scenario(scenario):
				if np.max(obs_data[Y]) > 1: 
					ATE = est_general.estimate_case_by_case(G, X, Y, y_val, obs_data, clip_val = False, cluster_variables = cluster_variables)
				else:
					ATE = est_general.estimate_case_by_case(G, X, Y, y_val, obs_data, cluster_variables = cluster_variables)
				# Else 
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
	ylim_var = kwargs.pop('ylim_var', None)

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

	plt.ylim(ylim_var)
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


def call_examples(example_number, **kwargs):
	d = kwargs.get('dim', None)
	if example_number == 1:
		# Back-door 
		scm, X, Y = example_SCM.Kang_Schafer(seednum = seednum)	
		example_name = 'Kang_Schafer'
		cluster_variables = []

	elif example_number == 11:
		# Back-door 		
		scm, X, Y = example_SCM.Kang_Schafer_dim(seednum = seednum, d=d)	
		example_name = f'Kang_Schafer_dim{d}'
		cluster_variables = ['Z']

	elif example_number == 12:
		# Back-door 		
		scm, X, Y = example_SCM.Kang_Schafer_dim(seednum = seednum, d=d)	
		example_name = f'Dukes_Vansteelandt_Farrel{d}'
		cluster_variables = ['Z']

	elif example_number == 2:
		# mSBD
		scm, X, Y = example_SCM.mSBD_SCM(seednum = seednum, d=d)	
		example_name = 'CanonMSBD'
		cluster_variables = ['Z1', 'Z2']

	elif example_number == 21:
		# mSBD
		scm, X, Y = example_SCM.Luedtke_v1(seednum = seednum)	
		example_name = 'Luedtke_v1'
		cluster_variables = []

	elif example_number == 22:
		# mSBD
		scm, X, Y = example_SCM.Luedtke_v2(seednum = seednum)	
		example_name = 'Luedtke_v2'
		cluster_variables = []

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

	elif example_number == 9: 
		# Double-FD-Ratio
		scm, X, Y = example_SCM.Napkin_FD_v2_SCM(seednum = seednum)
		example_name = 'CanonRatioFD2'

	return scm, X, Y, example_name, cluster_variables


def random_scm_experiments(seednum, **kwargs):
	# Random
	np.random.seed(seednum)
	random.seed(seednum)

	# Global Simulation 
	# num_sim = 4 
	# num_sample = 1000
	# simulation_round = 3 
	# cluster_variables = None 
	# scenario = 2

	num_sim = kwargs.get('num_sim', 4)
	list_of_samples = kwargs.get('list_of_samples', [100, 20000, 50000, 100000])
	# num_sample = kwargs.get('num_sample', 1000)
	simulation_round = kwargs.get('simulation_round', 3)
	cluster_variables = kwargs.get('cluster_variables', None)
	scenario = kwargs.get('scenario', 2)

	list_of_estimators = ['OM', 'IPW', 'DML']

	
	individual_simulation_counter = 0 

	scm_seednum_list = [random.randint(1, 1000000) for _ in range(num_sim)]
	sample_seednum_list = [random.randint(1, 1000000) for _ in range(simulation_round)]
	performance_dict = dict()

	for scm_seednum in scm_seednum_list:
		performance_dict[scm_seednum] = dict()
		for num_sample in list_num_samples:
			performance_dict[scm_seednum][num_sample] = dict()
			for estimator in list_of_estimators:
				performance_dict[scm_seednum][num_sample][estimator] = list()
			

	for scm_seednum in scm_seednum_list:
		num_observables = kwargs.get('num_observables', random.randint(5, 15))  # A random integer between 1 and 10
		num_unobservables = kwargs.get('num_unobservables', random.randint(5, num_observables))

		num_treatments = kwargs.get('num_treatments', random.randint(1, 5))
		num_outcomes = 1

		scm, X, Y = random_generator.Random_SCM_Generator2(num_observables = num_observables, num_unobservables = num_unobservables, num_treatments = num_treatments, num_outcomes = num_outcomes, condition_ID = True, seednum = scm_seednum)

		G = scm.graph
		G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
		observables = [node for node in G.nodes if not node.startswith('U')]
		y_val = np.ones(len(Y)).astype(int)
		truth = statmodules.ground_truth(scm, X, Y, y_val)

		for num_sample in list_num_samples:
			for sample_seednum in sample_seednum_list:
				individual_simulation_counter += 1 
				df_SCM = scm.generate_samples(num_sample, seed=sample_seednum)
				obs_data = df_SCM[observables]

				with simulate_scenario(scenario):
					if np.max(obs_data[Y]) > 1: 
						ATE = est_general.estimate_case_by_case(G, X, Y, y_val, obs_data, clip_val = False, cluster_variables = cluster_variables)
					else:
						ATE = est_general.estimate_case_by_case(G, X, Y, y_val, obs_data, cluster_variables = cluster_variables)
					# Else 
					_, _, performance_dict_per_seed, _ = statmodules.compute_performance(truth, ATE)

					for estimator in list_of_estimators: 
						performance_dict[scm_seednum][num_sample][estimator].append( performance_dict_per_seed[estimator] )

				print( "Progress:", np.round( (individual_simulation_counter/(num_sim * simulation_round * len(list_num_samples))) * 100, 3 ))
	
	return performance_dict

if __name__ == "__main__":
	''' 
	===== Kang_Schafer =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 1 241007 1200 1
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 1 241007 1200 1
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 1 241007 1200 1
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 1 241007 1200 1

	''' 
	===== Kang_Schafer dimensional (d=10) =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 11 241007 1200 50
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 11 241007 1200 50
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 11 241007 1200 50
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 11 241007 1200 50

	''' 
	===== Dukes_Vansteelandt_Farrel (d=20) =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 12 241007 1200 100
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 12 241007 1200 100
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 12 241007 1200 100
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 12 241007 1200 100


	''' 
	===== mSBD =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 2 241007 1500 50
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 2 241007 1500 50
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 2 241007 1500 50
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 2 241007 1500 50

	''' 
	===== Luedtke_v1 =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 21 241007 1500 1
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 21 241007 1500 1
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 21 241007 1500 1
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 21 241007 1500 1

	''' 
	===== Luedtke_v2 =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 22 241007 2100 1
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 22 241007 2100 1
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 22 241007 2100 1
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 22 241007 2100 1

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


	''' 
	===== Ratio FD2 =====
	'''
	''' scenario 1 '''
	# python3 simulation.py 190702 100 1 9 240917 0000 
	''' scenario 2 '''
	# python3 simulation.py 190702 100 2 9 240917 0000 
	''' scenario 3 '''
	# python3 simulation.py 190702 100 3 9 240917 0000 
	''' scenario 4 '''
	# python3 simulation.py 190702 100 4 9 240917 0000 


	''' 
	===== External (vs. Internal) Variable =====
	'''

	# seednum = int(sys.argv[1])
	# simulation_round = int(sys.argv[2])
	# scenario = int(sys.argv[3])
	# example_number = int(sys.argv[4])
	# sim_date = sys.argv[5]
	# sim_time = sys.argv[6]
	# sim_dim = int(sys.argv[7])

	seednum = 190702
	simulation_round = 10
	scenario = 3
	example_number = 21
	sim_date = 241118
	sim_time = 1500
	sim_dim = 4


	''' 
	===== Simulation ON (Fixed Graph) =====
	'''

	# np.random.seed(seednum)
	# random.seed(seednum)

	# list_num_samples = [100, 20000, 50000, 100000]
	# # list_num_samples = [100, 1000, 10000]
	# list_of_estimators = ['OM', 'IPW', 'DML']
	
	# scm, X, Y, example_name, cluster_variables = call_examples(example_number, dim=sim_dim)

	# pkl_path = 'log_experiments/pkl/'
	# fig_path = 'log_experiments/plot/'

	# filename = f'{sim_date}{sim_time}_{example_name}_seednum{seednum}_scenario{scenario}_round{simulation_round}'

	# print(f'base_seed: {seednum}, simulation round: {simulation_round}, scenario: {scenario}, example_number: {example_number}, example_name: {example_name}, sim_date_time: {sim_date}_{sim_time}')

	# pkl_extension = '.pkl'
	# fig_extension = '.png'

	# pkl_filename = filename + pkl_extension
	# fig_filename = filename + fig_extension
	# fig_size = (12,8)

	# performance_dict = run_DML_simulation(simulation_round, list_num_samples, list_of_estimators, scenario, seednum, scm, X, Y, pkl_path, pkl_filename, cluster_variables = cluster_variables)

	# if scenario == 1:
	# 	legend_on = True
	# else:
	# 	legend_on = False

	# draw_plots(performance_dict, fig_size = fig_size, fig_path = fig_path, fig_filename = fig_filename, fontsize_xtick = 30, fontsize_ytick = 30, legend_on = legend_on)


	''' 
	===== Random Simulation  =====
	'''

	seednum = 190702
	num_sim = 10
	simulation_round = 10
	list_num_samples = [100, 20000, 50000, 100000]
	performance_dict = random_scm_experiments(seednum = seednum, num_sim = num_sim, simulation_round = simulation_round, list_num_samples = list_num_samples)

