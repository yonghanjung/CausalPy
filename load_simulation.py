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

def readSimulation(param_file_name, result_file_name):
	param_file_name = "log_experiments/pkl/" + param_file_name
	with open(param_file_name, 'rb') as param_file:
		param = pickle.load(param_file)

	# Dynamically create variables
	for key, value in param.items():
		globals()[key] = value

	result_file_name = "log_experiments/pkl/" + result_file_name
	with open(result_file_name, 'rb') as result_file:
		performance_dict = pickle.load(result_file)    

	# Load Plot 
	fig_size = (10,12)
	fig_path = None 
	legend_on = True
	ylim_var = (0,0.5)

	draw_plots(performance_dict, fig_size = fig_size, fig_path = fig_path, fontsize_xtick = 40, fontsize_ytick = 40, legend_on = legend_on, ylim_var = ylim_var)


if __name__ == "__main__":
	param_file_name = "parameters_2409170_CanonDoubleNapkin_seednum190702_scenario2_round100.pkl.pkl"
	result_file_name = "result_2409170_CanonDoubleNapkin_seednum190702_scenario2_round100.pkl.pkl"

	readSimulation(param_file_name, result_file_name)


	# fig_size = (8,12)
	# custom_xticks = True or False 
	# custom_yticks = [0.05, 0.1, 0.15, 0.2] or None
	# list_num_samples = List of sample sizes
	# list_of_estimators = List of estimators to plot



