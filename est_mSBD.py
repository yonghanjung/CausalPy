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
                    obs_test_temp = obs_test.copy()
                    obs_test_temp['mu_xZ'] = mu_xZ
                    obs_test_temp['mu_XZ'] = mu_XZ_preds[test_index]
                    pi_XZ = statmodules.quadratic_balancing(
                        obs=obs_test_temp, x_val=x_val_row.values, X=X, Z=Z_unfold,
                        col_feature_1='mu_xZ', col_feature_2='mu_XZ'
                    )
                    pi_XZ_preds[x_val_tuple][test_index] = pi_XZ
        
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








def estimate_mSBD_xval_yval(G, X, Y, xval, yval, obs_data, alpha_CI = 0.05, seednum = 123, only_OM = False, **kwargs): 
    """
    Estimate causal effects using the mSBD method.

    Parameters:
    G : Causal graph structure.
    X : List of treatment variables.
    Y : List of outcome variables.
    xval : List of values corresponding to X.
    yval : List of values corresponding to Y.
    obs_data : Observed data (Pandas DataFrame).
    alpha_CI : Confidence level for interval estimates (default 0.05).
    estimators : Method for estimation (default "DML").
    seednum : Random seed for reproducibility (default 123).

    Returns:
    ATE : Estimated average treatment effect.
    VAR : Variance of the estimate.
    lower_CI : Lower confidence interval of the estimate.
    upper_CI : Upper confidence interval of the estimate.
    """

    cluster_variables = kwargs.get('cluster_variables', None)

    np.random.seed(int(seednum))
    random.seed(int(seednum))

    # G_new = graph.unfold_graph_from_data(G, cluster_variables, obs_data)

    # Sort Y and yval according to the topological order of the graph
    topo_V = graph.find_topological_order(G)
    sorted_pairs = sorted(zip(Y, yval), key=lambda pair: topo_V.index(pair[0]))
    sorted_variables, sorted_values = zip(*sorted_pairs)
    Y = list(sorted_variables)
    yval = list(sorted_values)
    dict_yval = {Y[idx]: yval[idx] for idx in range(len(Y))}

    # Check for SAC criterion satisfaction and organize variables into dictionaries
    dict_X, dict_Z, dict_Y = mSBD.check_SAC_with_results(G,X,Y, minimum = True)
    X_list = list(tuple(dict_X.values()))
    mSBD_length = len(dict_X)


    # Compute IyY: indicator for the outcome variables matching yval
    IyY = ((obs_data[Y] == tuple(yval))*1).prod(axis=1)
    obs_data_y = obs_data[:]
    obs_data_y.loc[:, 'IyY'] = np.asarray(IyY)

    # Create additional indicators for conditional variables
    for idx, (key, value) in enumerate(dict_Y.items()):
        if len(value) > 0:
            list_dict_yval = [dict_yval[value_iter] for value_iter in value]
            obs_data_y.loc[:, f'IyY_{idx}'] = ((obs_data[value] == list_dict_yval).all(axis=1)*1)

    m = len(dict_X)
    z_score = norm.ppf(1 - alpha_CI / 2)

    ATE = {}
    VAR = {}
    lower_CI = {}
    upper_CI = {}

    list_estimators = ["OM"] if only_OM else ["OM", "IPW", "DML"]

    for estimator in list_estimators:
        ATE[estimator] = 0
        VAR[estimator] = 0
        lower_CI[estimator] = 0
        upper_CI[estimator] = 0

    all_Z = []
    for each_Z_list in list(tuple(dict_Z.values())):
        all_Z += each_Z_list

    X_values_combinations = pd.DataFrame(product(*[np.unique(obs_data[Xi]) for Xi in X]), columns=X)

    # No confounding variables, simple estimation
    if not all_Z:
        for estimator in list_estimators:
            mask = (obs_data_y[X] == xval).all(axis=1) 
            ATE[estimator] = obs_data_y.loc[mask]['IyY'].mean()
            VAR[estimator] = obs_data_y.loc[mask]['IyY'].var()

    # Confounding variables present, use KFold cross-validation
    else:
        L = 2 # Number of folds 
        kf = KFold(n_splits=L, shuffle=True)

        mu_models = {}
        mu_eval_test_dict = {}
        check_mu_train_dict = {}
        check_mu_test_dict = {}

        pi_eval_dict = {}

        for train_index, test_index in kf.split(obs_data_y):
            obs_train, obs_test = obs_data_y.iloc[train_index], obs_data_y.iloc[test_index]
            check_mu_train_dict[m+1] = obs_train['IyY'].values
            check_mu_test_dict[m+1] = obs_test['IyY'].values

            # Loop through layers in reverse order
            for i in range(m, 0, -1):
                col_feature = []
                for j in range(1,i+1):
                    col_feature += dict_X[f'X{j}']
                    col_feature += dict_Z[f'Z{j}']
                for j in range(i):
                    col_feature += dict_Y[f'Y{j}']
                col_feature = sorted(col_feature, key=lambda x: topo_V.index(x))
                
                # Label for the current layer
                if i == m:
                    col_label = f'IyY_{m}'
                else:
                    col_label = f'check_mu_{i+1}'
                
                # Train model for the current layer
                mu_models[i] = statmodules.learn_mu(obs_train, col_feature, col_label, params=None)
                mu_eval_test_dict[i] = xgb_predict(mu_models[i], obs_test, col_feature)
                # mu_eval_test_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_test[col_feature]))
                for j in range(i):
                    key_j = f'IyY_{j}'
                    if key_j not in obs_data_y: 
                        continue 
                    else:
                        mu_eval_test_dict[i] *= obs_test[key_j]
                obs_test.loc[:,f'mu_{i}'] = mu_eval_test_dict[i]
                
                # Prepare train and test sets for the next iteration
                obs_test_x = copy.copy(obs_test)
                obs_test_x[dict_X[f'X{i}'][0]] = xval[X.index(dict_X[f'X{i}'][0])]
                obs_train_x = copy.copy(obs_train)
                obs_train_x[dict_X[f'X{i}'][0]] = xval[X.index(dict_X[f'X{i}'][0])]

                check_mu_train_dict[i] = xgb_predict(mu_models[i], obs_train_x, col_feature)
                # check_mu_train_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_train_x[col_feature]))
                for j in range(i):
                    key_j = f'IyY_{j}'
                    if key_j not in obs_data_y: 
                        continue 
                    else:
                        check_mu_train_dict[i] *= obs_train[key_j]
                obs_train.loc[:, f'check_mu_{i}'] = check_mu_train_dict[i]

                check_mu_test_dict[i] = xgb_predict(mu_models[i], obs_test_x, col_feature)
                # check_mu_test_dict[i] = mu_models[i].predict(xgb.DMatrix(obs_test_x[col_feature]))
                for j in range(i):
                    key_j = f'IyY_{j}'
                    if key_j not in obs_data_y: 
                        continue 
                    else:
                        check_mu_test_dict[i] *= obs_test[key_j]
                obs_test.loc[:, f'check_mu_{i}'] = check_mu_test_dict[i]

                # Compute weights for entropy balancing (if not only outcome model)
                if only_OM == False:
                    if i == 1 and len(dict_Y['Y0']) == 0 and len(dict_Z['Z1']) == 0: 
                        IxiX = (obs_test[dict_X[f'X{i}'][0]].values == xval[X.index(dict_X[f'X{i}'][0])]) * 1
                        P_X1_1 = np.mean(obs_test[dict_X[f'X{i}'][0]].values)
                        P_X1 = P_X1_1 * obs_test[dict_X[f'X{i}'][0]].values + (1-P_X1_1) * (1-obs_test[dict_X[f'X{i}'][0]].values )
                        pi_XZ = IxiX/P_X1

                    else:
                        pi_XZ = statmodules.entropy_balancing_osqp(obs = obs_test, 
                                                                x_val = xval[X.index(dict_X[f'X{i}'][0])], 
                                                                X = dict_X[f'X{i}'], 
                                                                Z = list(set(col_feature) - set(dict_X[f'X{i}'])), 
                                                                col_feature_1 = f'check_mu_{i}', 
                                                                col_feature_2 = f'mu_{i}')
                    
                    pi_eval_dict[i] = pi_XZ

            # Outcome model 
            if only_OM:
                OM_val = np.mean(obs_test['check_mu_1'])
                ATE["OM"] += OM_val
                VAR["OM"] += np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )

            # Double machine learning (DML), Outcome model (OM) and inverse probability weighting (IPW)
            else:
                pseudo_outcome = np.zeros(len(pi_eval_dict[m]))
                pi_accumulated_dict = {}
                pi_accumulated = np.ones(len(pi_eval_dict[m]))
                for i in range(1,m+1):
                    pi_accumulated_dict[i] = pi_accumulated * pi_eval_dict[i]
                    pi_accumulated *= pi_eval_dict[i]

                for i in range(m, 0, -1):
                    pseudo_outcome += pi_accumulated_dict[i] * (check_mu_test_dict[i+1] - mu_eval_test_dict[i])
                pseudo_outcome += check_mu_test_dict[i]

                OM_val = np.mean(obs_test['check_mu_1'])
                IPW_val = np.mean(pi_accumulated_dict[m] * check_mu_test_dict[m+1])
                AIPW_val = np.mean(pseudo_outcome)

                ATE["OM"] += OM_val
                VAR["OM"] += np.mean( (obs_test['check_mu_1'] - OM_val) ** 2 )
                
                ATE["DML"] += AIPW_val
                VAR["DML"] += np.mean( (pseudo_outcome - AIPW_val) ** 2 )

                ATE["IPW"] += IPW_val
                VAR["IPW"] += np.mean( (pi_accumulated_dict[m] * check_mu_test_dict[m+1] - IPW_val) ** 2 )
        
        for estimator in list_estimators:
            ATE[estimator] /= L
            VAR[estimator] /= L

    for estimator in list_estimators:
        mean_ATE = ATE[estimator]
        lower_x = (mean_ATE - z_score * VAR[estimator] * (len(obs_data_y) ** (-1/2)) )
        upper_x = (mean_ATE + z_score * VAR[estimator] * (len(obs_data_y) ** (-1/2)) )
        lower_CI[estimator] = lower_x
        upper_CI[estimator] = upper_x
    
    return ATE, VAR, lower_CI, upper_CI




if __name__ == "__main__":
    # Generate random SCM and preprocess the graph
    seednum = int(time.time())

    print(f'Random seed: {seednum}')
    np.random.seed(seednum)
    random.seed(seednum)
 
    d = 5
    scm, X, Y = example_SCM.mSBD_SCM_JCI(seednum,d)
    # scm, X, Y = example_SCM.BD_SCM(seednum,d)
    # scm, X, Y = example_SCM.luedtke_2017_sim1_scm(seednum)
    # scm, X, Y = example_SCM.Kang_Schafer(seednum)
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

    y_val = np.ones(len(Y)).astype(int)
    truth = statmodules.ground_truth(scm, X, Y, y_val)

    start_time = time.process_time()
    # ATE, VAR, lower_CI, upper_CI = estimate_BD(G, X, Y, obs_data, alpha_CI = 0.05, cluster_map = cluster_map)
    ATE, VAR, lower_CI, upper_CI = estimate_SBD(G, X, Y, obs_data, alpha_CI = 0.05, cluster_map = cluster_map)
    end_time = time.process_time()
    print(f'Time with OSQP minimizer: {end_time - start_time}')

    performance_table, rank_correlation_table, performance, rank_correlation_pvalue = statmodules.compute_performance(truth, ATE)
    
    print("Performance")
    print(performance_table)

    print("Rank Correlation")
    print(rank_correlation_table)

    

