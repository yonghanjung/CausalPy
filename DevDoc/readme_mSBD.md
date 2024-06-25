This document provides a detailed explanation of a Python implementation for various causal inference functions related to the modified Sequential Back-Door (mSBD) criterion and Sequential Admissibility (SAC) criterion. These functions help identify causal paths, check causal criteria, and construct adjustment sets in directed graphs using the NetworkX library.

# Table of Contents

1. [Introduction](#Introduction)
2. [Setup](#Setup)
3. [Modules](#Modules)
    1. [Partition Y](#Partition-Y)
    2. [Check mSBD Criterion](#Check-mSBD-Criterion)
    3. [Construct mSBD Z](#Construct-mSBD-Z)
    4. [Constructive mSBD Criterion](#Constructive-mSBD-Criterion)
    5. [Check mSBD with Results](#Check-mSBD-with-Results)
    6. [Check SAC Criterion](#Check-SAC-Criterion)
    7. [Construct SAC Z](#Construct-SAC-Z)
    8. [Construct Minimum SAC Z](#Construct-Minimum-SAC-Z)
    9. [Constructive SAC Criterion](#Constructive-SAC-Criterion)
    10. [Check SAC with Results](#Check-SAC-with-Results)
    11. [mSBD Estimand](#mSBD-Estimand)
4. [Example Usage](#Example-Usage)

# Introduction

This code document covers the implementation of several functions to handle causal inference tasks using NetworkX for graph manipulations. These tasks include identifying proper causal paths, checking the mSBD and SAC criteria, and constructing adjustment sets for estimating causal effects.

# Setup

Ensure you have the necessary libraries installed:

```bash
pip install networkx
```

Then, import the required libraries:

```python
import networkx as nx
import graph  # Assuming `graph` is a custom module containing auxiliary functions
import adjustment  # Assuming `adjustment` is a custom module containing additional causal inference functions
```

# Modules

## Partition Y

The `partition_Y` function partitions the set Y based on the descendants of nodes in X.

```python
def partition_Y(G, X, Y):
    X_topo_sorted = list(nx.topological_sort(G))
    X = [x for x in X_topo_sorted if x in X]

    partition = dict()
    all_descendants_of_X = graph.find_descendant(G, X)
    partition['Y0'] = list(set(Y) - set(all_descendants_of_X))

    for i, Xi in enumerate(X):
        so_far_Y = list()
        for j in range(i+1):
            so_far_Y += partition[f'Y{i}']
        remaining_Y = list(set(Y) - set(so_far_Y))
        descendants_of_Xi = graph.find_descendant(G, [Xi])
        if i < len(X) - 1:
            descendants_of_next_X = graph.find_descendant(G, X[i+1:])
            partition[f'Y{i+1}'] = list(set(remaining_Y).intersection(set(descendants_of_Xi) - set(descendants_of_next_X)))
        else:
            partition[f'Y{i+1}'] = list(set(remaining_Y).intersection(set(descendants_of_Xi)))
    return partition
```

## Check mSBD Criterion

The `check_mSBD_criterion_fixed_order` function checks the mSBD criterion with a fixed topological order of X.

```python
def check_mSBD_criterion_fixed_order(G, X, Y, Z):
    Y_partitions = partition_Y(G, X, Y)
    for i, Xi in enumerate(X):
        Zi = Z.get(f'Z{i+1}', [])
        if any(node in graph.find_descendant(G, X[i:]) for node in Zi):
            return False

        G_oi = graph.G_cut_incoming_edges(graph.G_cut_outgoing_edges(G, [Xi]), X[i+1:])
        past_Z = list(set().union(*[Z[f'Z{j}'] for j in range(1, i+1)]))
        past_Y = list(set().union(*[Y_partitions[f'Y{j}'] for j in range(1, i+1)]))
        past_X = list(set(X) - set(X[i:]))
        history_i = (past_Z + past_Y + past_X)
        future_Y = list(set(Y) - set(past_Y))
        conditioning_set = history_i + Zi

        if not graph.is_d_separated(G_oi, [Xi], future_Y, conditioning_set):
            return False

    return True
```

## Construct mSBD Z

The `construct_mSBD_Z` function constructs the candidate Z for checking mSBD.

```python
def construct_mSBD_Z(G, X, Y):
    X = list(X)
    Y_partitions = partition_Y(G, X, Y)
    Z = dict()
    m = max([int(key[1:]) for key in Y_partitions.keys()])
    for i, Xi in enumerate(X):
        G_oi = graph.G_cut_incoming_edges(graph.G_cut_outgoing_edges(G, [Xi]), X[i+1:])
        past_Z = list(set().union(*[Z[f'Z{j}'] for j in range(1, i+1)]))
        past_Y = list(set().union(*[Y_partitions[f'Y{j}'] for j in range(1, i+1)]))
        past_X = list(set(X) - set(X[i:]))
        history_i = (past_Z + past_Y + past_X)

        De_X_i = graph.find_descendant(G, X[i:])
        Forbidden_list = list(set(X) | set(Y) | set(history_i) | set(De_X_i))

        future_Y = list(set(Y) - set(past_Y))
        Zi = list(set(graph.find_ancestor(G_oi, [Xi] + future_Y + history_i)) - set(Forbidden_list))
        Z[f'Z{i+1}'] = Zi

    return Z
```

## Constructive mSBD Criterion

The `constructive_mSBD_criterion_fixed_order` function checks if Z satisfies the modified Sequential Back-Door (mSBD) criterion in graph G.

```python
def constructive_mSBD_criterion_fixed_order(G, X, Y):
    return check_mSBD_criterion_fixed_order(G, X, Y, construct_mSBD_Z(G, X, Y))
```

## Constructive mSBD Criterion with All Possible Orders

The `constructive_mSBD_criterion` function checks if Z satisfies the mSBD criterion for any topological order of X.

```python
def constructive_mSBD_criterion(G, X, Y):
    X_list = graph.all_possible_orders_X(G, X)
    for X_order in X_list:
        if constructive_mSBD_criterion_fixed_order(G, X_order, Y):
            return True
    return False
```

## Check mSBD with Results

The `check_mSBD_with_results` function checks if P(Y | do(X)) can be represented as an mSBD, and if so, provides the partitioned X, Z, Y.

```python
def check_mSBD_with_results(G, X, Y):
    if not constructive_mSBD_criterion(G, X, Y):
        raise ValueError("Not mSBD Admissible")

    X_list = graph.all_possible_orders_X(G, X)
    for X_order in X_list:
        if constructive_mSBD_criterion(G, X_order, Y):
            break

    m = len(X_order)
    dict_X = {}
    for idx in range(len(X_order)):
        dict_X[f"X{idx+1}"] = {X_order[idx]}

    dict_Y = partition_Y(G, X_order, Y)
    dict_Z = construct_mSBD_Z(G, X_order, Y)

    for i in reversed(range(m)):
        idx = i + 1
        if len(dict_Y[f"Y{idx}"]) > 0:
            break
        else:
            del dict_Y[f"Y{idx}"]
            del dict_Z[f"Z{idx}"]
            del dict_X[f"X{idx}"]

    return [dict_X, dict_Z, dict_Y]
```

## Check SAC Criterion

The `check_SAC_criterion_fixed_order` function checks the SAC criterion with a fixed topological order of X.

```python
def check_SAC_criterion_fixed_order(G, X, Y, Z):
    Y_partitions = partition_Y(G, X, Y)
    for i, Xi in enumerate(X):
        past_Z = list(set().union(*[Z[f'Z{j}'] for j in range(1, i+1)]))
        past_Y = list(set().union(*[Y_partitions[f'Y{j}'] for j in range(0, i+1)]))
        past_X = list(set(X) - set(X[i:]))
        history_i = (past_Z + past_Y + past_X)
        future_Y = list(set(Y) - set(past_Y))
        Zi = Z.get(f'Z{i+1}', [])
        if any(node in graph.find_descendant(G, X[i+1:]) for node in Zi):
            return False

        if any(node in adjustment.descedent_proper_causal_path(G, [Xi], future_Y) for node in Zi):
            return False

        G_psbd_i = adjustment.pro

per_backdoor_graph(graph.G_cut_incoming_edges(G, X[i+1:]), [Xi], future_Y)
        conditioning_set = history_i + Zi

        if not graph.is_d_separated(G_psbd_i, [Xi], future_Y, conditioning_set):
            return False

    return True
```

## Construct SAC Z

The `construct_SAC_Z` function constructs the candidate Z for checking SAC.

```python
def construct_SAC_Z(G, X, Y):
    X = list(X)
    Y_partitions = partition_Y(G, X, Y)
    Z = dict()
    m = max([int(key[1:]) for key in Y_partitions.keys()])
    for i, Xi in enumerate(X):
        past_Z = list(set().union(*[Z[f'Z{j}'] for j in range(1, i+1)]))
        past_Y = list(set().union(*[Y_partitions[f'Y{j}'] for j in range(1, i+1)]))
        past_X = list(set(X) - set(X[i:]))
        history_i = (past_Z + past_Y + past_X)
        future_Y = list(set(Y) - set(past_Y))

        G_psbd_i = adjustment.proper_backdoor_graph(graph.G_cut_incoming_edges(G, X[i+1:]), [Xi], future_Y)
        dpcp_i = adjustment.descedent_proper_causal_path(G, [Xi], future_Y)
        De_X_i1 = graph.find_descendant(G, X[i+1:])

        Forbidden_list = list(set(X) | set(Y) | set(history_i) | set(De_X_i1) | set(dpcp_i))
        Zi = list(set(graph.find_ancestor(G_psbd_i, [Xi] + future_Y + history_i)) - set(Forbidden_list))
        Z[f'Z{i+1}'] = Zi
    return Z
```

## Construct Minimum SAC Z

The `construct_minimum_SAC_Z` function constructs the minimal candidate Z for checking SAC.

```python
def construct_minimum_SAC_Z(G, X, Y):
    Y_partitions = partition_Y(G, X, Y)
    Z = construct_SAC_Z(G, X, Y)
    m = max([int(key[1:]) for key in Y_partitions.keys()])
    Zmin = dict()
    for i, Xi in enumerate(X):
        past_Z = list(set().union(*[Zmin[f'Z{j}'] for j in range(1, i+1)]))
        past_Y = list(set().union(*[Y_partitions[f'Y{j}'] for j in range(1, i+1)]))
        past_X = list(set(X) - set(X[i:]))
        history_i = (past_Z + past_Y + past_X)
        future_Y = list(set(Y) - set(past_Y))

        G_psbd_i = adjustment.proper_backdoor_graph(graph.G_cut_incoming_edges(G, X[i+1:]), [Xi], future_Y)
        Zi = Z[f'Z{i+1}']
        reacheable_Y = graph.find_reacheable_set(G_psbd_i, future_Y, Zi + history_i, Zi + history_i)
        Zi_1 = list(set(Zi).intersection(set(reacheable_Y)))
        reacheable_X = graph.find_reacheable_set(G_psbd_i, [Xi], Zi + history_i, Zi_1 + history_i)
        Zi_min = list(set(Zi_1).intersection(set(reacheable_X)))

        Zmin[f'Z{i+1}'] = Zi_min
    return Zmin
```

## Constructive SAC Criterion

The `constructive_SAC_criterion_fixed_order` function checks if Z satisfies the SAC criterion in graph G.

```python
def constructive_SAC_criterion_fixed_order(G, X, Y):
    return check_SAC_criterion_fixed_order(G, X, Y, construct_SAC_Z(G, X, Y))
```

## Constructive SAC Criterion with All Possible Orders

The `constructive_SAC_criterion` function checks if Z satisfies the SAC criterion for any topological order of X.

```python
def constructive_SAC_criterion(G, X, Y):
    X_list = graph.all_possible_orders_X(G, X)
    for X_order in X_list:
        if constructive_SAC_criterion_fixed_order(G, X_order, Y):
            return True
    return False
```

## Check SAC with Results

The `check_SAC_with_results` function checks if P(Y | do(X)) can be represented as a sequential admissible, and if so, provides the partitioned X, Z, Y.

```python
def check_SAC_with_results(G, X, Y, minimum=False):
    if not constructive_SAC_criterion(G, X, Y):
        raise ValueError("Not Sequential Covariate Admissible")

    X_list = graph.all_possible_orders_X(G, X)
    for X_order in X_list:
        if constructive_SAC_criterion(G, X_order, Y):
            break

    m = len(X_order)
    dict_X = {}
    for idx in range(len(X_order)):
        dict_X[f"X{idx+1}"] = [X_order[idx]]

    dict_Y = partition_Y(G, X_order, Y)

    if minimum:
        dict_Z = construct_minimum_SAC_Z(G, X_order, Y)
    else:
        dict_Z = construct_SAC_Z(G, X_order, Y)

    for i in reversed(range(m)):
        idx = i + 1
        if len(dict_Y[f"Y{idx}"]) > 0:
            break
        else:
            del dict_Y[f"Y{idx}"]
            del dict_Z[f"Z{idx}"]
            del dict_X[f"X{idx}"]

    return [dict_X, dict_Z, dict_Y]
```

## mSBD Estimand

The `mSBD_estimand` function provides the estimand for the mSBD adjustment.

```python
def mSBD_estimand(G, X, Y, latex=False, minimum=False):
    if adjustment.check_admissibility(G, X, Y):
        if minimum:
            Z = adjustment.construct_minimum_adjustment_set(G, X, Y)
        else:
            Z = adjustment.construct_adjustment_set(G, X, Y)
        return adjustment.adjustment_estimand(X, Y, Z, latex)

    dict_X, dict_Z, dict_Y = check_SAC_with_results(G, X, Y, minimum)
    m = len(dict_X)

    dict_X["X0"] = list()
    dict_Z["Z0"] = list()

    dict_H = {f"H0": dict_X[f"X{0}"] + dict_Y[f"Y{0}"] + dict_Z[f"Z{0}"]}
    for i in range(1, m):
        dict_H[f"H{i}"] = dict_X[f"X{i}"] + dict_Y[f"Y{i}"] + dict_Z[f"Z{i}"] + dict_H[f"H{i-1}"]

    term_list = []
    for i in range(m):
        idx = i + 1
        Xi_1 = dict_X.get(f"X{i}", list())
        Yi_1 = dict_Y.get(f"Y{i}", list())
        Zi = dict_Z.get(f"Z{idx}", list())
        Zi_1 = dict_Z.get(f"Z{i}", list())

        Yi_1_Zi = Yi_1 + Zi
        given_terms = dict_H.get(f"H{i-1}", list()) + Xi_1 + Zi_1

        if len(Yi_1_Zi) > 0:
            if not latex:
                term = f"P({', '.join(Yi_1_Zi)}" + (f" | {', '.join(given_terms)}" if given_terms else "") + ")"
            else:
                term = f"P({', '.join(Yi_1_Zi)}" + (f" \\mid {', '.join(given_terms)}" if given_terms else "") + ")"
        else:
            continue
        term_list.append(term)

    Ym = dict_Y[f"Y{m}"]
    given_term_m = dict_H[f"H{m-1}"] + dict_X[f"X{m}"] + dict_Z[f"Z{m}"]
    if len(Ym) > 0:
        if not latex:
            term_list.append(f"P({', '.join(Ym)}" + (f" | {', '.join(given_term_m)}" if given_term_m else "") + ")")
        else:
            term_list.append(f"P({', '.join(Ym)}" + (f" \\mid {', '.join(given_term_m)}" if given_term_m else "") + ")")

    summands = {z.lower() for values in dict_Z.values() for z in values}
    summands_str = ', '.join(summands)
    term_list_expression = ' '.join(reversed(term_list))

    if len(summands) > 0:
        if not latex:
            final_estimand = f"\u03A3_{{{summands_str}}} {term_list_expression}"
        else:
            final_estimand = f"\\sum_{{{summands_str}}} {term_list_expression}"
    else:
        final_estimand = f"{term_list_expression}"

    return final_estimand
```

# Example Usage

Below is an example usage of the provided functions:

```python
if __name__ == "__main__":
    # Example graph
    G =

 nx.DiGraph()
    G.add_edges_from([
        ('X1', 'Y1'), ('X1', 'Z1'), ('Z1', 'Y1'),
        ('X2', 'Y2'), ('X2', 'Z2'), ('Z2', 'Y2'),
        ('U1', 'X1'), ('U1', 'Z1'), ('U2', 'X2'), ('U2', 'Z2')
    ])

    X = ['X1', 'X2']
    Y = ['Y1', 'Y2']

    # Partition Y
    partitioned_Y = partition_Y(G, X, Y)
    print("Partitioned Y:", partitioned_Y)

    # Check mSBD criterion
    Z = construct_mSBD_Z(G, X, Y)
    is_mSBD = check_mSBD_criterion_fixed_order(G, X, Y, Z)
    print("mSBD Criterion Satisfied:", is_mSBD)

    # Check SAC criterion
    Z = construct_SAC_Z(G, X, Y)
    is_SAC = check_SAC_criterion_fixed_order(G, X, Y, Z)
    print("SAC Criterion Satisfied:", is_SAC)

    # Get mSBD estimand
    estimand = mSBD_estimand(G, X, Y, latex=False)
    print("mSBD Estimand:", estimand)
```

This document provides an overview and usage of various causal inference functions related to the mSBD and SAC criteria. For further details and customization, please refer to the function definitions and comments within the code.