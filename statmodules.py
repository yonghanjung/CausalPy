import random 
import numpy as np
from scipy.special import expit
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize
from scipy.stats import norm
import copy
from scipy import stats
from itertools import product
from tabulate import tabulate
import graph 
from scipy.stats import spearmanr
from scipy.stats import norm

import osqp
from scipy import sparse
from typing import Any, Union, Sequence


def ground_truth(scm, X, Y, yval):
	def randomized_equation(**args):
		num_samples = args.pop('num_sample')
		return np.random.binomial(1, 0.5, num_samples)
	# Update SCM equations with randomized equations for each Xi in X

	G = scm.graph
	topo_V = graph.find_topological_order(G)

	truth = {}
	for Xi in X:
		scm.equations[Xi] = randomized_equation

	df_SCM = scm.generate_samples(1000000)
	observables = [node for node in df_SCM.columns if not node.startswith('U')]

	intv_data = df_SCM[observables]
	X_values_combinations = pd.DataFrame(product(*[np.unique(intv_data[Xi]) for Xi in X]), columns=X)

	if len(Y) == 1:
		for _, x_val in X_values_combinations.iterrows():
			mask = (intv_data[X] == x_val.values).all(axis=1)
			truth[tuple(x_val)] = intv_data.loc[mask, Y].mean().iloc[0]
		return truth 

	else:
		IyY = ((intv_data[Y] == tuple(yval))*1).prod(axis=1)
		intv_data_y = intv_data[:]
		intv_data_y.loc[:, 'IyY'] = np.asarray(IyY)
		for _, x_val in X_values_combinations.iterrows():
			mask = (intv_data_y[X] == x_val.values).all(axis=1)
			truth[tuple(x_val)] = intv_data_y.loc[mask, 'IyY'].mean()
		return truth 

def quadratic_balancing(obs, x_val, X, Z, col_feature_1='mu_xZ', col_feature_2='mu_XZ'):
    # Get data and identify the target subgroup
    # Note: This line assumes X is a list of columns, as in the original code.
    IxX = np.array((obs[X] == x_val).prod(axis=1))
    n = len(obs)
    mu_xZ = obs[col_feature_1].values
    mu_XZ = obs[col_feature_2].values

    # Find the indices and count of the target subgroup to solve a smaller problem
    idx_treated = np.where(IxX == 1)[0]
    m = len(idx_treated)
    if m == 0:
        raise ValueError("No observations found for the specified `x_val`.")

    # --- Setup the Quadratic Program for the m-sized treated group ---

    # Objective function: minimize (1/2) * w'w
    P = sparse.diags([1.0] * m, format='csc')
    q = np.zeros(m)

    # --- Constraints ---

    # Constraint 1: sum(w_i) = n
    A1 = sparse.csr_matrix(np.ones((1, m)))
    l1 = np.array([n])
    u1 = np.array([n])

    # Constraint 2: sum(w_i * moment_i) = sum(target_moment)
    Cval1 = mu_XZ[idx_treated]
    Cval2_sum = np.sum(mu_xZ)
    A2 = sparse.csr_matrix(Cval1.reshape(1, -1))
    l2 = np.array([Cval2_sum])
    u2 = np.array([Cval2_sum])
    
    # Constraint 3: Non-negativity (w_i >= 1e-5)
    # This is the key logical fix, applied correctly.
    A3 = sparse.eye(m, format='csc')
    l3 = np.full(m, 1e-5)
    u3 = np.full(m, np.inf)
    
    # Combine constraints
    A = sparse.vstack([A1, A2, A3], format='csc')
    l = np.hstack([l1, l2, l3])
    u = np.hstack([u1, u2, u3])
    
    # OSQP solver setup
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    # Solve the problem
    res = prob.solve()

    # Check the status of the solution
    if res.info.status != 'solved':
        raise RuntimeError(f"OSQP failed to find a solution. Status: {res.info.status}")
    
    # Place the optimized weights for the subgroup into a full-length vector
    w_opt_full = np.zeros(n)
    w_opt_full[idx_treated] = res.x

    return w_opt_full

# Function to compute the confidence interval
def mean_confidence_interval(data, confidence=0.95):
	data = np.array(data)
	mean = np.mean(data)
	sem = stats.sem(data)
	margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
	return mean, margin_of_error

# Extract error bars
def extract_error_bars(data):
	means = data
	errors = ci_acc
	return means, errors

def add_noise(vec,add_noise_TF):
	if add_noise_TF:
		n = len(vec)
		noise = np.random.normal(loc=n**(-1/4), scale=n**(-1/4), size=len(vec))
		vec += noise
	return vec

def add_noise_val(val, n, add_noise_TF):
	if add_noise_TF:
		noise = np.random.normal(loc=n**(-1/4), scale=n**(-1/4), size=1)
		val += noise
	return val

def learn_mu(obs, col_feature, col_label, params = None):
	# XGBoost regression model to regress Y on X and Z
	dtrain = xgb.DMatrix(obs[col_feature], label=obs[col_label])
	if params == None: 
		params = {
			'booster': 'gbtree',
			'eta': 0.3,
			'gamma': 0,
			'max_depth': 10,
			'min_child_weight': 1,
			'subsample': 1.0,
			'colsample_bytree': 1,
			'lambda': 0.0,
			'alpha': 0.0,
			'objective': 'reg:squarederror',
			'eval_metric': 'rmse',
			'n_jobs': 4  # Assuming you have 4 cores
		}
	bst = xgb.train(params, dtrain)
	return bst

def learn_pi(obs, col_feature, col_label, params=None):
	# XGBoost classification model to regress X on Z
	dtrain = xgb.DMatrix(obs[col_feature], label=obs[col_label])
	if params == None:
		params = {
			'booster': 'gbtree',
			'eta': 0.5,
			'gamma': 0,
			'max_depth': 20,
			'min_child_weight': 1,
			'subsample': 0.0,
			'colsample_bytree': 1,
			'objective': 'binary:logistic',  # Change as per your objective
			'eval_metric': 'logloss',  # Change as per your needs
			'reg_lambda': 0.0,
			'reg_alpha': 0.0,
			'nthread': 4
		}

	bst = xgb.train(params, dtrain)
	return bst

def learn_multi_pi(obs, col_feature, col_label, params=None):
	# XGBoost classification model to regress X on Z
	dtrain = xgb.DMatrix(obs[col_feature], label=obs[col_label])
	if params == None:
		params = {
			'booster': 'gbtree',
			'eta': 0.5,
			'gamma': 0,
			'max_depth': 20,
			'min_child_weight': 1,
			'subsample': 0.0,
			'colsample_bytree': 1,
			'objective': 'multi:softprob',  # Change as per your objective
			'num_class': len(np.unique(obs[col_label])),
			'eval_metric': 'softprob',  # Change as per your needs
			'reg_lambda': 0.0,
			'reg_alpha': 0.0,
			'nthread': 4
		}

	bst = xgb.train(params, dtrain)
	return bst

def find_mu_param(obs):
	features = [col for col in obs.columns if col not in ['Y']]
	# fixed_params = {
	# 	'booster': 'gbtree',
	# 	# 'eta': 0.5,
	# 	'gamma': 0,
	# 	# 'max_depth': 10,
	# 	'min_child_weight': 1,
	# 	'subsample': 0.8,
	# 	'colsample_bytree': 1,
	# 	'lambda': 0,
	# 	'alpha': 0,
	# 	'objective': 'reg:squarederror',
	# 	'eval_metric': 'rmse',
	# 	'n_jobs': 4  # Assuming you have 4 cores
	# }
	xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_jobs=4, booser = 'gbtree', gamma = 0, min_child_weight=1, subsample = 0.8, alpha=0)

	# Define the parameter grid
	param_grid = {
		'eta': [0.1, 0.3, 0.5, 1],
		'max_depth': [6, 10, 15]
	}

	# Initialize GridSearchCV
	grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=2, verbose=1, n_jobs=-1)

	# Fit the model
	grid_search.fit(obs[features], obs['Y'])

	return grid_search.best_params_

def estimate_odds_ratio(data_0, data_1, col_feature, n_sample, params = None):
	# Step 2: Randomly sample n_sample data points from both datasets
	samples_0 = data_0.sample(n=n_sample, random_state=42)
	samples_1 = data_1.sample(n=n_sample, random_state=42)
	
	# Step 3: Create a new dataframe with labels
	samples_0['L'] = 0
	samples_1['L'] = 1
	col_label = ['L']
	total_features = col_feature + col_label
	total_samples = pd.concat([samples_0[total_features], samples_1[total_features]], axis=0)
	
	# Step 4: Construct the XGBoost model
	model = learn_pi(total_samples, col_feature, col_label, params)
	return model

def compute_performance(truth, ATE):
	performance = {}
	rank_correlation_pvalue = {}

	for estimator in list(ATE.keys()):
		performance[estimator] = np.mean(np.abs(np.array(list(truth.values())) - np.array(list(ATE[estimator].values()))))
		rank_correlation_pvalue[estimator] = list( spearmanr(list(truth.values()), list(ATE[estimator].values())) )
	
	performance_table_data = [[estimator] + [performance[estimator]] for estimator in performance]
	performance_table_header = ["Estimator", "Error"]
	performance_table = tabulate(performance_table_data, tablefmt='grid', floatfmt=".3f", headers = performance_table_header)

	rank_correlation_table_data = [[estimator] + [value for value in rank_correlation_pvalue[estimator]] for estimator in rank_correlation_pvalue]
	rank_correlation_table_header = ["Estimator", "Rank Correlation", "P-value"]
	rank_correlation_table = tabulate(rank_correlation_table_data, tablefmt='grid', floatfmt=".3f", headers = rank_correlation_table_header)

	return performance_table, rank_correlation_table, performance, rank_correlation_pvalue
