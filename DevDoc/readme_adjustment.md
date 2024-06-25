# Code Document: Understanding and Using Causal Inference Functions

This document provides a detailed explanation of a Python implementation for various causal inference functions. These functions help identify causal paths, check causal criteria, and construct adjustment sets in directed graphs using the NetworkX library.

## Table of Contents

1. [Introduction](#Introduction)
2. [Setup](#Setup)
3. [Modules](#Modules)
    1. [Proper Causal Path](#Proper-Causal-Path)
    2. [Descendent Proper Causal Path](#Descendent-Proper-Causal-Path)
    3. [Backdoor Criterion](#Backdoor-Criterion)
    4. [Proper Backdoor Graph](#Proper-Backdoor-Graph)
    5. [Adjustment Criterion](#Adjustment-Criterion)
    6. [Construct Adjustment Set](#Construct-Adjustment-Set)
    7. [Admissibility](#Admissibility)
    8. [Adjustment Estimand](#Adjustment-Estimand)
    9. [Minimum Adjustment Set](#Minimum-Adjustment-Set)
4. [Example Usage](#Example-Usage)

## Introduction

This code document covers the implementation of several functions to handle causal inference tasks using NetworkX for graph manipulations. These tasks include identifying proper causal paths, checking the backdoor and adjustment criteria, and constructing adjustment sets for estimating causal effects.

## Setup

Ensure you have the necessary libraries installed:

```bash
pip install networkx
```

Then, import the required libraries:

```python
import networkx as nx
import graph  # Assuming `graph` is a custom module containing auxiliary functions
```

## Modules

### Proper Causal Path

The `proper_causal_path` function identifies nodes that are descendants of `X` (excluding `X`) and ancestors of `Y`.

```python
def proper_causal_path(G, X, Y):
    # Step 1: Remove incoming edges to X
    Gi = graph.G_cut_incoming_edges(G, X)

    # Step 2: Remove outgoing edges from X
    Go = graph.G_cut_outgoing_edges(G, X)

    # Step 3: Find descendants of X in Gi
    de_X = graph.find_descendant(Gi, X)

    # Step 4: Find ancestors of Y in Go
    an_Y = graph.find_ancestor(Go, Y)

    # Step 5: Return the intersection, excluding X
    return list((set(de_X) - set(X)) & set(an_Y))
```

### Descendent Proper Causal Path

The `descedent_proper_causal_path` function finds all descendants of nodes identified by `proper_causal_path`.

```python
def descedent_proper_causal_path(G, X, Y):
    pcp = proper_causal_path(G, X, Y)
    return graph.find_descendant(G, pcp)
```

### Backdoor Criterion

The `check_backdoor_criterion` function checks if a set `Z` satisfies the Back-door Criterion relative to `X` and `Y`.

```python
def check_backdoor_criterion(G, X, Y, Z):
    descendants_of_X = graph.find_descendant(G, X)
    if any(z in descendants_of_X for z in Z):
        return False

    G_modified = graph.G_cut_outgoing_edges(G, X)
    if graph.is_d_separated(G_modified, X, Y, Z):
        return True

    return False
```

### Proper Backdoor Graph

The `proper_backdoor_graph` function removes edges from nodes in `X` to nodes identified by `proper_causal_path`.

```python
def proper_backdoor_graph(G, X, Y):
    X = list(X)
    Y = list(Y)

    pcp = proper_causal_path(G, X, Y)
    
    G_modified = G.copy()

    for x_node in X:
        for y_node in pcp:
            if G_modified.has_edge(x_node, y_node):
                G_modified.remove_edge(x_node, y_node)

    return G_modified
```

### Adjustment Criterion

The `check_adjustment_criterion` function checks if a set `Z` satisfies the adjustment criterion relative to `(X, Y)`.

```python
def check_adjustment_criterion(G, X, Y, Z):
    G_pbd = proper_backdoor_graph(G, X, Y)
    dpcp = descedent_proper_causal_path(G, X, Y)
    if any(z in dpcp for z in Z):
        return False

    if graph.is_d_separated(G_pbd, X, Y, Z):
        return True

    return False
```

### Construct Adjustment Set

The `construct_adjustment_set` function constructs an adjustment set for estimating the causal effect of `X` on `Y`.

```python
def construct_adjustment_set(G, X, Y):
    X_set = set(X)
    Y_set = set(Y)
    dpcp = descedent_proper_causal_path(G, X, Y)
    dpcp_set = set(dpcp)

    ancestors_XY = graph.find_ancestor(G, X_set.union(Y_set))
    ancestors_XY_set = set(ancestors_XY)

    adjustment_set = ancestors_XY_set - (X_set.union(Y_set).union(dpcp_set))
    return list(adjustment_set)
```

### Admissibility

The `check_admissibility` function checks if `P(Y | do(X))` can be represented as a back-door adjustment.

```python
def check_admissibility(G, X, Y):
    adjustment_Z = construct_adjustment_set(G, X, Y)
    if check_adjustment_criterion(G, X, Y, adjustment_Z):
        return True
    return False
```

### Adjustment Estimand

The `adjustment_estimand` function generates the back-door adjustment formula in either plain text or LaTeX format.

```python
def adjustment_estimand(X, Y, Z, latex):
    Z = list(set(Z))
    Z_val = ', '.join(Z)
    Z_lower_val = ', '.join(char.lower() for char in Z)

    Y_val = ', '.join(Y)
    X_val = ', '.join(X)
    XZ = list(set(X).union(set(Z)))
    XZ_val = ', '.join(XZ)

    if not latex:
        if len(Z) == 0:
            adjustment_estimand = f"P({Y_val} | {X_val})"
        else:
            adjustment_estimand = f"\u03A3_{{{Z_lower_val}}}P({Y_val} | {XZ_val}) P({Z_val})"
    else:
        if len(Z) == 0:
            adjustment_estimand = f"P({Y_val} \\mid {X_val})"
        else:
            adjustment_estimand = f"\\sum_{{{Z_lower_val}}}P({Y_val} \\mid {XZ_val}) P({Z_val})"
    return adjustment_estimand
```

### Minimum Adjustment Set

The `construct_minimum_adjustment_set` function constructs a minimum adjustment set for estimating the causal effect of `X` on `Y`.

```python
def construct_minimum_adjustment_set(G, X, Y):
    if check_adjustment_criterion(G, X, Y, []):
        return set([])
    Z = construct_adjustment_set(G, X, Y)
    reacheable_Y = graph.find_reacheable_set(G, Y, Z, Z)
    Z1 = list(set(Z).intersection(set(reacheable_Y)))
    reacheable_X = graph.find_reacheable_set(G, X, Z, reacheable_Y)
    Z2 = list(set(Z1).intersection(set(reacheable_X)))
    return Z2
```

## Example Usage

Below is an example usage of the provided functions:

```python
if __name__ == "__main__":
    # Example graph
    G = nx.DiGraph()
    G.add_edges_from([
        ('X', 'Y'), ('X', 'Z'), ('Z', 'Y'),
        ('U', 'X'), ('U', 'Z')
    ])

    X = ['X']
    Y = ['Y']

    # Find proper causal path
    pcp = proper_causal_path(G, X, Y)
    print("Proper Causal Path:", pcp)

    # Check backdoor criterion
    Z = ['Z']
    is_backdoor = check_backdoor_criterion(G, X, Y, Z)
    print("Backdoor Criterion Satisfied:", is_backdoor)

    # Check adjustment criterion
    is_adjustment = check_adjustment_criterion(G, X, Y, Z)
    print("Adjustment Criterion Satisfied:", is_adjustment)

    # Construct adjustment set
    adj_set = construct_adjustment_set(G, X, Y)
    print("Adjustment Set:", adj_set)

    # Check admissibility
    is_admissible = check_admissibility(G, X, Y)
    print("Admissible for Back-door Adjustment:", is_admissible)

    # Generate adjustment estimand
    estimand = adjustment_estimand(X, Y, Z, latex=False)
    print("Adjustment Estimand:", estimand)
```

This document provides an overview and usage of various causal inference functions. For further details and customization, please refer to the function definitions and comments within the code.