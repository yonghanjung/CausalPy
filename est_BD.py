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



if __name__ == "__main__":
	# Generate random SCM and preprocess the graph
	seednum = int(time.time())
	# seednum = 1725560261

	print(f'Random seed: {seednum}')
	np.random.seed(seednum)
	random.seed(seednum)

	scm, X, Y = random_generator.random_SCM_generator(
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
	ATE, VAR, lower_CI, upper_CI = estimate_BD(G, X, Y, obs_data, alpha_CI = 0.05, seednum = 123, only_OM = False)
	end_time = time.process_time()
	print(f'Time with SciPy minimizer: {end_time - start_time}')
	performance_table, rank_correlation_table = statmodules.compute_performance(truth, ATE)

	print("Performance")
	print(performance_table)

	print("Rank Correlation")
	print(rank_correlation_table)

	# start_time = time.process_time()
	# ATE, VAR, lower_CI, upper_CI = estimate_BD_osqp(G, X, Y, obs_data, alpha_CI = 0.05, seednum = 123, only_OM = False)
	# end_time = time.process_time()
	# print(f'Time with OSQP: {end_time - start_time}')

	performance_table, rank_correlation_table = statmodules.compute_performance(truth, ATE)

	print("Performance")
	print(performance_table)

	print("Rank Correlation")
	print(rank_correlation_table)

	

	

	


