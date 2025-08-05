import numpy as np
import pandas as pd
from itertools import product
from itertools import chain
from sklearn.model_selection import KFold
import xgboost as xgb
import copy
from scipy.stats import spearmanr
from scipy.stats import norm
import warnings
import random 
import time 
from tabulate import tabulate

import SCM 
import random_generator
import graph
import identify
import adjustment
import frontdoor
import mSBD
import tian
import statmodules
import example_SCM

# Turn off alarms
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")

# Suppress all UserWarning messages globally from the osqp package
warnings.filterwarnings("ignore", category=UserWarning, module='osqp')

def add_cluster_to_vec(cluster_variables, vec):
    vec_new = []
    if cluster_variables is not None:
        for node in vec:
            if node in cluster_variables:
                # Expand cluster variable: find columns starting with the node, excluding the node itself
                expanded = [col for col in obs_data.columns if col.startswith(node) and col != node]
                # Sort the expanded columns to ensure proper ordering (e.g., C1, C2, C3)
                expanded_sorted = sorted(expanded, key=lambda x: (len(x), x))
                vec_new.extend(expanded_sorted)
            else:
                vec_new.append(node)
    return vec_new

def xgb_predict(model, data, col_feature):
    return model.predict(xgb.DMatrix(data[col_feature]))

def estimate_BD(G, X, Y, obs_data, alpha_CI=0.05, cluster_map=None, n_folds=2, seednum=123, only_OM=False):
    """
    Estimates causal effects using the backdoor adjustment criterion.

    This function identifies the minimum adjustment set Z and computes the average
    treatment effect (ATE) using Double Machine Learning (DML) with cross-fitting.
    It provides estimates for Outcome Model (OM), Inverse Probability Weighting (IPW),
    and Doubly Robust (DML/AIPW) estimators.

    Parameters:
    G : Causal graph structure.
    X : List of treatment variables.
    Y : List of outcome variables.
    obs_data : Observed data (Pandas DataFrame).
    alpha_CI : Confidence level for interval estimates (default 0.05).
    cluster_map : Dictionary mapping conceptual nodes to column names (default None).
    n_folds : Number of folds for cross-fitting (default 2).
    seednum : Random seed for reproducibility (default 123).
    only_OM : If True, only computes the Outcome Model estimate (default False).

    Returns:
    A tuple containing dictionaries for ATE, Variance, Lower CI, and Upper CI.
    """
    # 1. --- Initial Setup ---
    np.random.seed(int(seednum))
    random.seed(int(seednum))

    Z = adjustment.construct_minimum_adjustment_set(G, X, Y)
    Z_unfold = graph.expand_variables(Z, cluster_map)
    X_unfold = graph.expand_variables(X, cluster_map)
    X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Xi]) for Xi in X]), columns=X)

    list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]
    results = {est: {'ATE': {}, 'VAR': {}} for est in list_estimators}

    # 2. --- Estimation Logic ---
    if not Z:
        # Case 1: No confounding, simple stratified estimation
        for _, x_val_row in X_values_combinations.iterrows():
            x_val = tuple(x_val_row)
            mask = (obs_data[X] == x_val_row.values).all(axis=1)
            mean_val = obs_data.loc[mask, Y].mean().iloc[0]
            var_val = obs_data.loc[mask, Y].var().iloc[0]
            for est in list_estimators:
                results[est]['ATE'][x_val] = mean_val
                results[est]['VAR'][x_val] = var_val
    else:
        # Case 2: Confounding, use DML with cross-fitting
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(seednum))
        col_feature = list(set(Z_unfold + X_unfold))
        
        mu_XZ_preds = np.zeros(len(obs_data))
        mu_xZ_preds = {tuple(x_val): np.zeros(len(obs_data)) for _, x_val in X_values_combinations.iterrows()}
        pi_XZ_preds = {tuple(x_val): np.zeros(len(obs_data)) for _, x_val in X_values_combinations.iterrows()}

        for train_index, test_index in kf.split(obs_data):
            obs_train, obs_test = obs_data.iloc[train_index], obs_data.iloc[test_index]
            nuisance_mu = statmodules.learn_mu(obs_train, col_feature, Y, params=None)
            mu_XZ_preds[test_index] = xgb_predict(nuisance_mu, obs_test, col_feature)

            for _, x_val_row in X_values_combinations.iterrows():
                x_val_tuple = tuple(x_val_row)
                obs_test_x = obs_test.copy()
                obs_test_x[X] = x_val_row.values
                
                mu_xZ = xgb_predict(nuisance_mu, obs_test_x, col_feature)
                mu_xZ_preds[x_val_tuple][test_index] = mu_xZ

                if not only_OM:
                    # --- Start of corrected block ---
                    n_test = len(obs_test)
                    # Identify the subgroup in the current test fold
                    subgroup_mask = (obs_test[X] == x_val_row.values).all(axis=1)
                    subgroup_indices = np.where(subgroup_mask)[0]
                    
                    # Initialize weights for this fold to zero
                    pi_XZ_fold = np.zeros(n_test)

                    if np.sum(subgroup_mask) > 0:
                        # Moment features are the E[Y|X,Z] predictions for the subgroup
                        moment_features = mu_XZ_preds[test_index][subgroup_mask]
                        # Target sum is the sum of E[Y|x,Z] predictions over the whole fold
                        target_moment_sum = np.sum(mu_xZ)
                        
                        try:
                            # Solve for the weights of the subgroup members
                            subgroup_weights = statmodules._solve_single_step_weights(
                                n=n_test,
                                subgroup_indices=subgroup_indices,
                                moment_features=moment_features,
                                target_moment_sum=target_moment_sum
                            )
                            # Place the solved weights into the full array for the fold
                            pi_XZ_fold[subgroup_indices] = subgroup_weights
                        except RuntimeError as e:
                            # Fallback if solver fails
                            print(f"Warning: OSQP solver failed ({e}). Using equal weights for subgroup {x_val_tuple}.")
                            equal_weight = n_test / len(subgroup_indices)
                            pi_XZ_fold[subgroup_indices] = equal_weight

                    pi_XZ_preds[x_val_tuple][test_index] = pi_XZ_fold
                    # --- End of corrected block ---

        
        # Compute final estimates for each estimator using a single loop
        Yvec = obs_data[Y].values.flatten()
        for _, x_val_row in X_values_combinations.iterrows():
            x_val = tuple(x_val_row)
            
            # Define the final "outcome" vector for each estimator
            estimator_outcomes = {
                'OM': mu_xZ_preds[x_val],
                'IPW': pi_XZ_preds[x_val] * Yvec,
                'DML': mu_xZ_preds[x_val] + pi_XZ_preds[x_val] * (Yvec - mu_XZ_preds)
            }
            
            # Loop through the desired estimators and calculate results
            for est in list_estimators:
                final_outcome_vec = estimator_outcomes[est]
                results[est]['ATE'][x_val] = np.mean(final_outcome_vec)
                results[est]['VAR'][x_val] = np.var(final_outcome_vec)

    # 3. --- Calculate Confidence Intervals ---
    ATE, VAR = {k: v['ATE'] for k, v in results.items()}, {k: v['VAR'] for k, v in results.items()}
    lower_CI, upper_CI = {est: {} for est in list_estimators}, {est: {} for est in list_estimators}
    z_score = norm.ppf(1 - alpha_CI / 2)
    n_samples = len(obs_data)

    for est in list_estimators:
        for x_val, ate_val in ATE[est].items():
            std_err = (VAR[est][x_val] / n_samples) ** 0.5
            lower_CI[est][x_val] = ate_val - z_score * std_err
            upper_CI[est][x_val] = ate_val + z_score * std_err

    return ATE, VAR, lower_CI, upper_CI

def estimate_SBD(G, X, Y, obs_data, alpha_CI=0.05, cluster_map=None, n_folds=2, seednum=123, only_OM=False):
    """
    Estimates causal effects using the sequential backdoor criterion (g-formula).

    This function implements a sequential estimation strategy based on the
    g-formula, providing estimates for the Outcome Model (OM), Inverse Probability
    Weighting (IPW), and Doubly Robust (DML/AIPW) estimators using
    cross-fitting. This version uses sequential_quadratic_balancing for weights.

    Parameters:
    G : Causal graph structure.
    X : List of treatment variables.
    Y : List of outcome variables.
    obs_data : Observed data (Pandas DataFrame).
    alpha_CI : Confidence level for interval estimates (default 0.05).
    cluster_map : Dictionary mapping conceptual nodes to column names (default None).
    n_folds : Number of folds for cross-fitting (default 2).
    seednum : Random seed for reproducibility (default 123).
    only_OM : If True, only computes the Outcome Model estimate (default False).

    Returns:
    A tuple containing dictionaries for ATE, Variance, Lower CI, and Upper CI.
    """
    # 1. --- Initial Setup ---
    np.random.seed(int(seednum))
    random.seed(int(seednum))

    topo_V = graph.find_topological_order(G)
    dict_X, dict_Z, _ = mSBD.check_SAC_with_results(G, X, Y, minimum=True)

    # Unfold the clustered variables
    reverse_cluster_map = {}
    if cluster_map:
        dict_Z = {k: graph.expand_variables(v, cluster_map) for k, v in dict_Z.items()}
        dict_X = {k: graph.expand_variables(v, cluster_map) for k, v in dict_X.items()}
        for cluster_name, members in cluster_map.items():
            for member in members:
                reverse_cluster_map[member] = cluster_name

    all_Z = sorted(list(set(chain.from_iterable(dict_Z.values()))))
    X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Xi]) for Xi in X]), columns=X)
    
    list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]
    results = {est: {'ATE': {}, 'VAR': {}} for est in list_estimators}
    m = len(dict_X)
    n_samples = len(obs_data)

    # 2. --- Estimation Logic ---
    if not all_Z:
        # Case 1: No confounding, simple stratified estimation
        for _, x_val_row in X_values_combinations.iterrows():
            x_val = tuple(x_val_row)
            mask = (obs_data[X] == x_val_row.values).all(axis=1)
            mean_val = obs_data.loc[mask, Y].mean().iloc[0]
            var_val = obs_data.loc[mask, Y].var().iloc[0]
            for est in list_estimators:
                results[est]['ATE'][x_val] = mean_val
                results[est]['VAR'][x_val] = var_val
    else:
        # --- Stage 1: Cross-fitting to estimate all moment functions (mu and check_mu) ---
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(seednum))

        mu_preds = {i: np.zeros(n_samples) for i in range(1, m + 1)}
        check_mu_preds = {}
        pi_preds = {} # Will be populated after moment estimation

        for _, x_val_row in X_values_combinations.iterrows():
            x_val_tuple = tuple(x_val_row)
            check_mu_preds[(m + 1, x_val_tuple)] = obs_data[Y].values.flatten()
            for i in range(1, m + 1):
                 check_mu_preds[(i, x_val_tuple)] = np.zeros(n_samples)

        for train_index, test_index in kf.split(obs_data):
            obs_train, obs_test = obs_data.iloc[train_index], obs_data.iloc[test_index]

            for _, x_val_row in X_values_combinations.iterrows():
                x_val_tuple = tuple(x_val_row)
                check_mu_train_fold = {m + 1: obs_train[Y].values.flatten()}

                for i in range(m, 0, -1):
                    col_feature_i = []
                    for j in range(1, i + 1):
                        col_feature_i.extend(dict_X[f'X{j}'])
                        col_feature_i.extend(dict_Z[f'Z{j}'])
                    
                    # def get_sort_key(node):
                    #     base_node = reverse_cluster_map.get(node, node)
                    #     return topo_V.index(base_node)
                    # col_feature_i = sorted(list(set(col_feature_i)), key=get_sort_key)
                    
                    label_data = check_mu_train_fold[i + 1]
                    label_col_name = f'pseudo_outcome_label_for_stage_{i+1}'
                    obs_train_temp = obs_train.copy()
                    obs_train_temp[label_col_name] = label_data
                    
                    nuisance_mu_i = statmodules.learn_mu(obs_train_temp, col_feature_i, label_col_name, params=None)
                    
                    mu_preds[i][test_index] = xgb_predict(nuisance_mu_i, obs_test, col_feature_i)
                    
                    obs_test_x = obs_test.copy()
                    obs_test_x[dict_X[f'X{i}'][0]] = x_val_row[dict_X[f'X{i}'][0]]
                    check_mu_preds[(i, x_val_tuple)][test_index] = xgb_predict(nuisance_mu_i, obs_test_x, col_feature_i)
                    
                    obs_train_x = obs_train.copy()
                    obs_train_x[dict_X[f'X{i}'][0]] = x_val_row[dict_X[f'X{i}'][0]]
                    check_mu_train_fold[i] = xgb_predict(nuisance_mu_i, obs_train_x, col_feature_i)

        # --- Stage 2: Compute Importance Weights Sequentially using full prediction vectors ---
        if not only_OM:
            for _, x_val_row in X_values_combinations.iterrows():
                x_val_tuple = tuple(x_val_row)
                
                # Prepare a dataframe with the original data and all necessary moment predictions
                obs_for_weights = obs_data.copy()
                mu_col_names = []
                check_mu_col_names = []

                for i in range(1, m + 1):
                    mu_col = f'mu_{i}_pred'
                    check_mu_col = f'check_mu_{i}_{x_val_tuple}_pred'
                    mu_col_names.append(mu_col)
                    check_mu_col_names.append(check_mu_col)
                    obs_for_weights[mu_col] = mu_preds[i]
                    obs_for_weights[check_mu_col] = check_mu_preds[(i, x_val_tuple)]

                # Call the sequential balancing function
                all_pi_weights_dict = statmodules.sequential_quadratic_balancing(
                    obs=obs_for_weights,
                    X_cols=X,
                    x_vals=list(x_val_tuple),
                    mu_cols=mu_col_names,
                    check_mu_cols=check_mu_col_names
                )

                # Store the results
                for i in range(1, m + 1):
                    pi_preds[(i, x_val_tuple)] = all_pi_weights_dict[f'pi_{i}']


        # --- Stage 3: Compute Final Estimates from Full Prediction Vectors ---
        Yvec = obs_data[Y].values.flatten()
        for _, x_val_row in X_values_combinations.iterrows():
            x_val = tuple(x_val_row)
            
            # Outcome Model (g-formula) estimate is always computed
            estimator_outcomes = {'OM': check_mu_preds[(1, x_val)]}
            
            if not only_OM:
                # IPW estimator
                pi_accumulated = np.ones(n_samples)
                for i in range(1, m + 1):
                    # The weights are non-zero only for the subgroup that received the treatment
                    # up to stage i. We multiply them sequentially.
                    pi_accumulated = pi_preds[(i, x_val)]
                
                # The final weights pi^m are non-zero only for the subgroup with X=x.
                # The product Y * pi^m correctly computes the weighted average for this subgroup.
                estimator_outcomes['IPW'] = pi_accumulated * Yvec

                # DML (AIPW) estimator
                pseudo_outcome_dml = np.zeros(n_samples)
                pi_acc_dict = {0: np.ones(n_samples)} # pi^0 = 1 for all
                for i in range(1, m + 1):
                    pi_acc_dict[i] = pi_preds[(i, x_val)]

                # Summation term: sum_{i=1 to m} pi^{i-1}*(check_mu^i - mu^i)
                # Note: The recursive formula simplifies this. The DML estimator for SBD is
                # sum_{i=1 to m} [pi^i - pi^{i-1}]E[Y|...]_i + pi^m * Y
                # A more direct implementation is the sequential DML pseudo-outcome:
                pseudo_outcome_dml = check_mu_preds[(1, x_val)].copy() # Starts with OM
                for i in range(1, m + 1):
                    # pi^{i-1} is implicitly used in the calculation of pi^i
                    # The DML update at stage i uses pi^i
                    pseudo_outcome_dml += pi_acc_dict[i] * (check_mu_preds[(i + 1, x_val)] - mu_preds[i])

                estimator_outcomes['DML'] = pseudo_outcome_dml

            for est in list_estimators:
                final_outcome_vec = estimator_outcomes[est]
                results[est]['ATE'][x_val] = np.mean(final_outcome_vec)
                results[est]['VAR'][x_val] = np.var(final_outcome_vec) / len(final_outcome_vec) # Use variance of the mean

    # 3. --- Calculate Confidence Intervals ---
    ATE, VAR = {k: v['ATE'] for k, v in results.items()}, {k: v['VAR'] for k, v in results.items()}
    lower_CI, upper_CI = {est: {} for est in list_estimators}, {est: {} for est in list_estimators}
    z_score = norm.ppf(1 - alpha_CI / 2)

    for est in list_estimators:
        for x_val, ate_val in ATE[est].items():
            std_err = (VAR[est][x_val]) ** 0.5
            lower_CI[est][x_val] = ate_val - z_score * std_err
            upper_CI[est][x_val] = ate_val + z_score * std_err

    return ATE, VAR, lower_CI, upper_CI


def estimate_mSBD_xy(G, X, Y, y_policy, obs_data, alpha_CI=0.05, cluster_map=None, n_folds=2, seednum=123, only_OM=False, verbose=False):
    """
    Estimates causal effects for time-varying treatments and outcomes.

    This function adapts the sequential backdoor logic for settings where
    outcomes Y_t can be confounders for subsequent treatments X_{t+1}. It
    estimates the probability of achieving a target final outcome Y_m=y_m,
    under a specific treatment policy `x` and intermediate outcome policy `y`.

    Parameters:
    G : Causal graph structure.
    X : List of treatment variable names.
    Y : List of outcome variable names.
    x_policy : Tuple representing the target treatment policy (e.g., (1, 0, 1)).
    y_policy : Tuple representing the target intermediate outcome policy (e.g., (1, 1, 0)).
    obs_data : Observed data (Pandas DataFrame).
    alpha_CI, cluster_map, n_folds, seednum, only_OM, verbose: Same as estimate_SBD.

    Returns:
    A tuple of dictionaries for ATE, Variance, and CIs for the single target policy.
    """
    # 1. --- Initial Setup ---
    np.random.seed(int(seednum))
    random.seed(int(seednum))

    dict_X, dict_Z, dict_Y = mSBD.check_SAC_with_results(G, X, Y, minimum=True)
    
    topo_V = graph.find_topological_order(G)
    dict_y = dict(zip(Y, y_policy))
    
    dict_Z = {f'Z{i}': list(set(dict_Z.get(f'Z{i}', []) + dict_Y.get(f'Y{i-1}', []))) for i in range(1, len(dict_Z) + 1)}
    
    
    IyY = obs_data[Y].eq(pd.Series(dict_y)).all(axis=1).astype(int)
    Y_label = 'Y'
    obs_data[Y_label] = IyY

    # Unfold the clustered variables
    reverse_cluster_map = {}
    if cluster_map:
        dict_Z = {k: graph.expand_variables(v, cluster_map) for k, v in dict_Z.items()}
        dict_X = {k: graph.expand_variables(v, cluster_map) for k, v in dict_X.items()}
        for cluster_name, members in cluster_map.items():
            for member in members:
                reverse_cluster_map[member] = cluster_name

    all_Z = sorted(list(set(chain.from_iterable(dict_Z.values()))))
    X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Xi]) for Xi in X]), columns=X)
    
    list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]
    results = {est: {'ATE': {}, 'VAR': {}} for est in list_estimators}
    m = len(dict_X)
    n_samples = len(obs_data)

    # 2. --- Estimation Logic ---
    if not all_Z:
        # Case 1: No confounding, simple stratified estimation
        for _, x_val_row in X_values_combinations.iterrows():
            x_val = tuple(x_val_row)
            mask = (obs_data[X] == x_val_row.values).all(axis=1)
            mean_val = obs_data.loc[mask, Y_label].mean().iloc[0]
            var_val = obs_data.loc[mask, Y_label].var().iloc[0]
            for est in list_estimators:
                results[est]['ATE'][x_val] = mean_val
                results[est]['VAR'][x_val] = var_val
    else:
        # --- Stage 1: Cross-fitting to estimate all moment functions (mu and check_mu) ---
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(seednum))

        mu_preds = {i: np.zeros(n_samples) for i in range(1, m + 1)}
        check_mu_preds = {}
        pi_preds = {} # Will be populated after moment estimation

        for _, x_val_row in X_values_combinations.iterrows():
            x_val_tuple = tuple(x_val_row)
            check_mu_preds[(m + 1, x_val_tuple)] = obs_data[Y_label].values.flatten()
            for i in range(1, m + 1):
                 check_mu_preds[(i, x_val_tuple)] = np.zeros(n_samples)

        for train_index, test_index in kf.split(obs_data):
            obs_train, obs_test = obs_data.iloc[train_index], obs_data.iloc[test_index]

            for _, x_val_row in X_values_combinations.iterrows():
                x_val_tuple = tuple(x_val_row)
                check_mu_train_fold = {m + 1: obs_train[Y_label].values.flatten()}

                for i in range(m, 0, -1):
                    col_feature_i = []
                    for j in range(1, i + 1):
                        col_feature_i.extend(dict_X[f'X{j}'])
                        col_feature_i.extend(dict_Z[f'Z{j}'])
                    
                    def get_sort_key(node):
                        base_node = reverse_cluster_map.get(node, node)
                        return topo_V.index(base_node)
                    col_feature_i = sorted(list(set(col_feature_i)), key=get_sort_key)
                    
                    label_data = check_mu_train_fold[i + 1]
                    label_col_name = f'pseudo_outcome_label_for_stage_{i+1}'
                    obs_train_temp = obs_train.copy()
                    obs_train_temp[label_col_name] = label_data
                    
                    nuisance_mu_i = statmodules.learn_mu(obs_train_temp, col_feature_i, label_col_name, params=None)
                    
                    mu_preds[i][test_index] = xgb_predict(nuisance_mu_i, obs_test, col_feature_i)
                    
                    obs_test_x = obs_test.copy()
                    obs_test_x[dict_X[f'X{i}'][0]] = x_val_row[dict_X[f'X{i}'][0]]
                    check_mu_preds[(i, x_val_tuple)][test_index] = xgb_predict(nuisance_mu_i, obs_test_x, col_feature_i)
                    
                    obs_train_x = obs_train.copy()
                    obs_train_x[dict_X[f'X{i}'][0]] = x_val_row[dict_X[f'X{i}'][0]]
                    check_mu_train_fold[i] = xgb_predict(nuisance_mu_i, obs_train_x, col_feature_i)

        # --- Stage 2: Compute Importance Weights Sequentially using full prediction vectors ---
        if not only_OM:
            for _, x_val_row in X_values_combinations.iterrows():
                x_val_tuple = tuple(x_val_row)
                
                # Prepare a dataframe with the original data and all necessary moment predictions
                obs_for_weights = obs_data.copy()
                mu_col_names = []
                check_mu_col_names = []

                for i in range(1, m + 1):
                    mu_col = f'mu_{i}_pred'
                    check_mu_col = f'check_mu_{i}_{x_val_tuple}_pred'
                    mu_col_names.append(mu_col)
                    check_mu_col_names.append(check_mu_col)
                    obs_for_weights[mu_col] = mu_preds[i]
                    obs_for_weights[check_mu_col] = check_mu_preds[(i, x_val_tuple)]

                # Call the sequential balancing function
                all_pi_weights_dict = statmodules.sequential_quadratic_balancing(
                    obs=obs_for_weights,
                    X_cols=X,
                    x_vals=list(x_val_tuple),
                    mu_cols=mu_col_names,
                    check_mu_cols=check_mu_col_names
                )

                # Store the results
                for i in range(1, m + 1):
                    pi_preds[(i, x_val_tuple)] = all_pi_weights_dict[f'pi_{i}']


        # --- Stage 3: Compute Final Estimates from Full Prediction Vectors ---
        Yvec = obs_data[Y_label].values.flatten()
        for _, x_val_row in X_values_combinations.iterrows():
            x_val = tuple(x_val_row)
            
            # Outcome Model (g-formula) estimate is always computed
            estimator_outcomes = {'OM': check_mu_preds[(1, x_val)]}
            
            if not only_OM:
                # IPW estimator
                pi_accumulated = np.ones(n_samples)
                for i in range(1, m + 1):
                    # The weights are non-zero only for the subgroup that received the treatment
                    # up to stage i. We multiply them sequentially.
                    pi_accumulated = pi_preds[(i, x_val)]
                
                # The final weights pi^m are non-zero only for the subgroup with X=x.
                # The product Y * pi^m correctly computes the weighted average for this subgroup.
                estimator_outcomes['IPW'] = pi_accumulated * Yvec

                # DML (AIPW) estimator
                pseudo_outcome_dml = np.zeros(n_samples)
                pi_acc_dict = {0: np.ones(n_samples)} # pi^0 = 1 for all
                for i in range(1, m + 1):
                    pi_acc_dict[i] = pi_preds[(i, x_val)]

                # Summation term: sum_{i=1 to m} pi^{i-1}*(check_mu^i - mu^i)
                # Note: The recursive formula simplifies this. The DML estimator for SBD is
                # sum_{i=1 to m} [pi^i - pi^{i-1}]E[Y|...]_i + pi^m * Y
                # A more direct implementation is the sequential DML pseudo-outcome:
                pseudo_outcome_dml = check_mu_preds[(1, x_val)].copy() # Starts with OM
                for i in range(1, m + 1):
                    # pi^{i-1} is implicitly used in the calculation of pi^i
                    # The DML update at stage i uses pi^i
                    pseudo_outcome_dml += pi_acc_dict[i] * (check_mu_preds[(i + 1, x_val)] - mu_preds[i])

                estimator_outcomes['DML'] = pseudo_outcome_dml

            for est in list_estimators:
                final_outcome_vec = estimator_outcomes[est]
                results[est]['ATE'][x_val] = np.mean(final_outcome_vec)
                results[est]['VAR'][x_val] = np.var(final_outcome_vec) / len(final_outcome_vec) # Use variance of the mean

    # 3. --- Calculate Confidence Intervals ---
    ATE, VAR = {k: v['ATE'] for k, v in results.items()}, {k: v['VAR'] for k, v in results.items()}
    lower_CI, upper_CI = {est: {} for est in list_estimators}, {est: {} for est in list_estimators}
    z_score = norm.ppf(1 - alpha_CI / 2)

    for est in list_estimators:
        for x_val, ate_val in ATE[est].items():
            std_err = (VAR[est][x_val]) ** 0.5
            lower_CI[est][x_val] = ate_val - z_score * std_err
            upper_CI[est][x_val] = ate_val + z_score * std_err

    return ATE, VAR, lower_CI, upper_CI




if __name__ == "__main__":
    # Generate random SCM and preprocess the graph
    seednum = int(time.time())

    print(f'Random seed: {seednum}')
    np.random.seed(seednum)
    random.seed(seednum)
 
    d = 4
    ''' BD '''
    # scm, X, Y = example_SCM.Kang_Schafer(seednum)
    # scm, X, Y = example_SCM.BD_SCM(seednum,d)
    
    ''' SBD '''
    # scm, X, Y = example_SCM.mSBD_SCM_JCI(seednum,d)
    # scm, X, Y = example_SCM.luedtke_2017_sim1_scm(seednum)
    
    ''' mSBD '''
    # scm, X, Y = example_SCM.mSBD_SCM(seednum,d)
    scm, X, Y = example_SCM.mSBD_SCM(seednum,d)
    G = scm.graph
    obs_data = scm.generate_observational_samples(100000)
 
    # 2. AUTOMATICALLY build the map
    cluster_map = graph.build_cluster_map(graph.find_topological_order(G),obs_data)
 
    # print("Automatically detected cluster map:")

    print( identify.causal_identification(G,X,Y, latex = False, copyTF=True) )
 
    # Check various criteria
    satisfied_BD = adjustment.check_admissibility(G, X, Y)
    satisfied_mSBD = mSBD.constructive_SAC_criterion(G, X, Y)
    satisfied_FD = frontdoor.constructive_FD(G, X, Y)
    satisfied_Tian = tian.check_Tian_criterion(G, X)
    satisfied_gTian = tian.check_Generalized_Tian_criterion(G, X)

    y_val = np.array([1,1])
    truth = statmodules.ground_truth(scm, X, Y, y_val)

    start_time = time.process_time()
    # ATE, VAR, lower_CI, upper_CI = estimate_SBD(G, X, Y, obs_data, alpha_CI = 0.05, cluster_map = cluster_map)
    ATE, VAR, lower_CI, upper_CI = estimate_mSBD_xy(G, X, Y, y_val, obs_data, alpha_CI = 0.05, cluster_map = cluster_map)
    end_time = time.process_time()
    print(f'Time with OSQP minimizer: {end_time - start_time}')

    performance_table, rank_correlation_table, performance, rank_correlation_pvalue = statmodules.compute_performance(truth, ATE)
    
    print("Performance")
    print(performance_table)

    print("Rank Correlation")
    print(rank_correlation_table)

    

