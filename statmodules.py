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
	intervened_scm = copy.deepcopy(scm)

	truth = {}
	for Xi in X:
		intervened_scm.equations[Xi] = randomized_equation

	df_SCM = intervened_scm.generate_samples(1_000_000)
	observables = [node for node in df_SCM.columns if not node.startswith('U')]

	intv_data = df_SCM[observables]
	X_values_combinations = pd.DataFrame(product(*[np.unique(intv_data[Xi]) for Xi in X]), columns=X)

	if len(Y) == 1:
		for _, x_val in X_values_combinations.iterrows():
			mask = (intv_data[X] == x_val.values).all(axis=1)
			truth[tuple(x_val)] = intv_data.loc[mask, Y].mean().iloc[0]
		return truth 

	else:
		# Name-aligned multi-Y indicator
		dict_y = dict(zip(Y, yval))
		IyY = intv_data[Y].eq(pd.Series(dict_y)).all(axis=1).astype(int)
		# Compute path probability by X cell
		for _, x_val in X_values_combinations.iterrows():
			mask = (intv_data[X] == x_val.values).all(axis=1)
			# Note: cast to float to avoid numpy types leaking out
			truth[tuple(x_val)] = float(IyY[mask].mean())
		return truth


def _solve_single_step_weights(n, subgroup_indices, moment_features, target_moment_sum, epsilon=0.0):
    """
    Core OSQP solver for a single time step.
    
    Args:
        ...
        epsilon (float): Slack allowed for the balancing constraint. A value of 0.0
                         enforces a strict equality.
    """
    m = len(subgroup_indices)
    if m == 0:
        return np.array([])

    # Objective function: minimize (1/2) * w'w
    P = sparse.diags([1.0] * m, format='csc')
    q = np.zeros(m)

    # --- Constraints ---
    # Constraint 1: sum(w_i) = n
    A1 = sparse.csr_matrix(np.ones((1, m)))
    l1 = np.array([n - (epsilon/target_moment_sum) * n])
    u1 = np.array([n + (epsilon/target_moment_sum) * n])

    # Constraint 2: sum(w_i * moment_feature_i) = target_moment_sum +/- epsilon
    A2 = sparse.csr_matrix(moment_features.reshape(1, -1))
    # [MODIFIED] Apply the epsilon buffer to the lower and upper bounds.
    l2 = np.array([target_moment_sum - epsilon])
    u2 = np.array([target_moment_sum + epsilon])

    # Constraint 3: Non-negativity (w_i >= 0)
    A3 = sparse.eye(m, format='csc')
    l3 = np.full(m, 1e-6) 
    u3 = np.full(m, np.inf)

    # Combine constraints
    A = sparse.vstack([A1, A2, A3], format='csc')
    l = np.hstack([l1, l2, l3])
    u = np.hstack([u1, u2, u3])

    # Setup and solve the QP problem
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=True)
    res = prob.solve()

    if res.info.status != 'solved':
        raise RuntimeError(f"OSQP failed to solve the QP. Status: {res.info.status}")

    return res.x


def sequential_quadratic_balancing(obs, X_cols, x_vals, mu_cols, check_mu_cols, verbose=False):
    """
    Calculates sequential weights with a robust retry mechanism and a verbose switch.
    """
    n = len(obs)
    num_steps = len(X_cols)

    if not all(len(lst) == num_steps for lst in [x_vals, mu_cols, check_mu_cols]):
        raise ValueError("Length of X_cols, x_vals, mu_cols, and check_mu_cols must be the same.")

    pi_previous = np.ones(n)
    all_weights = {}

    for i in range(num_steps):
        step = i + 1
        X_col, x_val = X_cols[i], x_vals[i]
        mu_col, check_mu_col = mu_cols[i], check_mu_cols[i]

        subgroup_mask = (obs[X_col] == x_val)
        subgroup_indices = np.where(subgroup_mask)[0]
        m = len(subgroup_indices)
        
        w_subgroup = np.array([])

        if m > 0:
            moment_features = obs.loc[subgroup_mask, mu_col].values
            check_mu_values = obs[check_mu_col].values
            target_moment_sum = np.sum(pi_previous * check_mu_values)
            
            # Robust Retry Logic
            try:
                # First attempt: strict equality constraint
                if verbose:
                    print(f"\n--- Step {step} for x_vals={x_vals}: Attempting strict solve (epsilon=0)...")
                w_subgroup = _solve_single_step_weights(
                    n, subgroup_indices, moment_features, target_moment_sum, epsilon=0.0
                )
                if verbose:
                    print("...Success: Strict solution found.")
            except RuntimeError as e:
                # Check if the error is the one we want to handle
                if "infeasible" in str(e).lower():
                    if verbose:
                        print(f"!!! Warning: Strict solve failed. Status: {e}")

                    # Retry with relaxed constraint
                    relaxation_factor = 0.05  # Allow 5% slack
                    base_epsilon = 1e-4 # Handles cases where target is near zero
                    epsilon = (relaxation_factor * np.abs(target_moment_sum)) + base_epsilon
                    
                    if verbose:
                        print(f"!!! Retrying with relaxed constraint (epsilon={epsilon:.4f})...")
                    
                    try:
                        w_subgroup = _solve_single_step_weights(
                            n, subgroup_indices, moment_features, target_moment_sum, epsilon=epsilon
                        )
                        if verbose:
                            print("...Success: Relaxed solution found.")
                    except RuntimeError as e2:
                        # If it fails even with relaxation, use the final fallback.
                        if verbose:
                            print(f"!!! CRITICAL: Relaxed solve also failed. Status: {e2}")
                            print("!!! Applying final fallback: Assigning equal weights.")
                        equal_weight = n / m
                        w_subgroup = np.full(m, equal_weight)
                else:
                    # It was a different runtime error, so we re-raise it.
                    raise e
        else:
            if verbose:
                print(f"\n--- Step {step} for x_vals={x_vals}: Subgroup is empty. Skipping.")

        # Place the solved (or fallback) weights into a full-length vector
        pi_current = np.zeros(n)
        if m > 0:
            pi_current[subgroup_indices] = w_subgroup
        
        all_weights[f'pi_{step}'] = pi_current
        pi_previous = pi_current
        
        # Verification prints
        if verbose:
            print(f"Sum of pi^{step}: {np.sum(pi_current):.4f} (should be approx. {n})")

    return all_weights


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
