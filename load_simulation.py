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

from simulation import *

def readSimulation(pkl_path, fig_path, filename):
	param_file_name = pkl_path + "parameters_" + filename + '.pkl'
	with open(param_file_name, 'rb') as param_file:
		param = pickle.load(param_file)

	# Dynamically create variables
	for key, value in param.items():
		globals()[key] = value

	result_file_name = pkl_path + "result_"  + filename + '.pkl'
	with open(result_file_name, 'rb') as result_file:
		performance_dict = pickle.load(result_file)    

	# Load Plot 
	fig_size = (8,8)
	legend_on = False
	ylim_var = (0,0.55)

	fig_filename = filename + '.png'

	draw_plots(performance_dict, fig_size = fig_size, fig_path = fig_path, fig_filename = fig_filename, fontsize_xtick = 25, fontsize_ytick = 40, legend_on = legend_on, ylim_var = ylim_var)


if __name__ == "__main__":

	seednum = int(sys.argv[1])
	simulation_round = int(sys.argv[2])
	scenario = int(sys.argv[3])
	example_number = int(sys.argv[4])
	sim_date = sys.argv[5]
	sim_time = sys.argv[6]
	sim_dim = int(sys.argv[7])

	# python3 load_simulation.py [seednum] [simulation_round] [scenario] [example_number] [sim_date] [sim_time] [sim_dim]

	##---------------## 
	## Fulcher FD 
	# python3 load_simulation.py 190702 100 2 31 250204 1230 999
	# python3 load_simulation.py 190702 100 3 31 250204 1230 999
	# python3 load_simulation.py 190702 100 4 31 250204 1230 999

	## PlanID
	# python3 load_simulation.py 190702 100 2 4 250204 1230 999
	# python3 load_simulation.py 190702 100 3 4 250204 1230 999
	# python3 load_simulation.py 190702 100 4 4 250204 1230 999

	## Napkin
	# python3 load_simulation.py 190702 100 2 5 250217 2200 999
	# python3 load_simulation.py 190702 100 3 5 250217 2200 999
	# python3 load_simulation.py 190702 100 4 5 250217 2200 999

	## NapkinFD
	# python3 load_simulation.py 190702 100 2 6 250204 1500 999
	# python3 load_simulation.py 190702 100 3 6 250204 1500 999
	# python3 load_simulation.py 190702 100 4 6 250204 1500 999

	## Tian Napkin
	# python3 load_simulation.py 190702 100 2 7 250204 1500 999
	# python3 load_simulation.py 190702 100 3 7 250204 1500 999
	# python3 load_simulation.py 190702 100 4 7 250204 1500 999

	## Tian Napkin
	# python3 load_simulation.py 190702 100 2 8 250204 1900 999
	# python3 load_simulation.py 190702 100 3 8 250204 1900 999
	# python3 load_simulation.py 190702 100 4 8 250204 1900 999

	## Double Napkin
	# python3 load_simulation.py 190702 100 2 8 250204 1900 999
	# python3 load_simulation.py 190702 100 3 8 250204 1900 999
	# python3 load_simulation.py 190702 100 4 8 250204 1900 999
	##---------------## 

	# seednum = 190702
	# simulation_round = 100
	# scenario = 1
	# example_number = 1
	# sim_date = 250123
	# sim_time = 1200
	# sim_dim = 10


	scm, X, Y, example_name, cluster_variables = call_examples(seednum, example_number, dim=sim_dim)

	pkl_path = 'log_experiments/pkl/'
	fig_path = 'log_experiments/plot/'

	filename = f'{sim_date}{sim_time}_{example_name}_seednum{seednum}_scenario{scenario}_round{simulation_round}'

	readSimulation(pkl_path, fig_path, filename)


	# fig_size = (8,12)
	# custom_xticks = True or False 
	# custom_yticks = [0.05, 0.1, 0.15, 0.2] or None
	# list_num_samples = List of sample sizes
	# list_of_estimators = List of estimators to plot



