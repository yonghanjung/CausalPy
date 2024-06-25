The `graph.py` module provides functions for creating, manipulating, and visualizing causal graphs. This document explains each function in detail, along with usage examples to help you understand how to use the module effectively.

# Table of Contents

1. [Introduction](#Introduction)
2. [Setup](#Setup)
3. [Modules](#Modules)
    1. [Visualize Graph](#Visualize-Graph)
    2. [Write Joint Distribution](#Write-Joint-Distribution)
    3. [Find Topological Order](#Find-Topological-Order)
    4. [Create Acyclic Graph](#Create-Acyclic-Graph)
    5. [Generate All Possible Orders of X](#Generate-All-Possible-Orders-of-X)
    6. [Create Subgraphs](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#create-subgraphs)
    7. [List All C-Components](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#list-all-c-components)
    8. [Find C-Components](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#find-c-components)
    9. [Find Parents](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#find-parents)
    10. [Find Descendants](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#find-descendants)
    11. [Find Ancestors](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#find-ancestors)
    12. [Subgraph of Ancestors of Y](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#subgraph-of-ancestors-of-y)
    13. [Check D-Separation](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#check-d-separation)
    14. [Cut Incoming Edges](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#cut-incoming-edges)
    15. [Cut Outgoing Edges](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#cut-outgoing-edges)
    16. [Generate Random Graph](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#generate-random-graph)
    17. [Check Inducing Paths](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#check-inducing-paths)
    18. [Check Inducing Paths with Unmeasured Variables](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#check-inducing-paths-with-unmeasured-variables)
    19. [Check if Node is Unmeasured](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#check-if-node-is-unmeasured)
    20. [Find Variables with No Inducing Paths](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#find-variables-with-no-inducing-paths)
    21. [Reachable Set](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#reachable-set)
    22. [Convert Graph Dictionary to Fusion Graph](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#convert-graph-dictionary-to-fusion-graph)
4. [Example Usage](notion://www.notion.so/yonghanjung/5f9f7537439744b1aac2530f1455e5d7?v=f8bb0fa24c20449f91681ba2cffff136&p=5f4f0ef10d8e42f9a4c0fbd5b39ba5d2&pm=s#example-usage)

---

# Introduction

The `graph.py` module is designed to assist with the creation and manipulation of causal graphs using the `networkx` library. The module includes functions for visualizing graphs, generating joint distributions, finding topological orders, creating acyclic graphs, and generating subgraphs.

# Setup

To use the `graph.py` module, ensure you have the following libraries installed:

```bash
pip install networkx matplotlib
```

# Modules

## Visualize Graph

**Function:** `visualize(graph)`

**Description:** Visualize the causal graph with colored nodes for treatments and outcomes.

**Parameters:**

- `graph` (nx.DiGraph): The graph to visualize.

**Usage Example:**

```python
import networkx as nx
from graph import visualize

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('T1', 'O2'), ('O1', 'Y')])
G.nodes['T1']['pos'] = (1, 2)
G.nodes['O1']['pos'] = (2, 3)
G.nodes['O2']['pos'] = (3, 1)
G.nodes['Y']['pos'] = (4, 2)

visualize(G)

```

## Write Joint Distribution

**Function:** `write_joint_distribution(variables)`

**Description:** Writes the joint probability distribution symbolically.

**Parameters:**

- `variables` (list): List of observed variables in topological order.

**Returns:**

- `str`: Symbolic representation of the joint probability distribution.

**Usage Example:**

```python
from graph import write_joint_distribution

variables = ['T1', 'O1', 'O2', 'Y']
joint_distribution = write_joint_distribution(variables)
print(joint_distribution)  # Output: P(T1 , O1 , O2 , Y)

```

## Find Topological Order

**Function:** `find_topological_order(G)`

**Description:** Finds the topological order of the observed variables in the graph.

**Parameters:**

- `G` (nx.DiGraph): The projected causal graph.

**Returns:**

- `list`: A list of observed variables in topological order.

**Usage Example:**

```python
import networkx as nx
from graph import find_topological_order

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('T1', 'O2'), ('O1', 'Y')])

topological_order = find_topological_order(G)
print(topological_order)  # Output: ['T1', 'O1', 'O2', 'Y']

```

## Create Acyclic Graph

**Function:** `create_acyclic_graph(graph_dict, an_Y_graph_TF=False, Y=None, node_positions=None)`

**Description:** Create a DAG from a dictionary representation, ensuring the graph is acyclic.

**Parameters:**

- `graph_dict` (dict): A dictionary representing the graph, where keys are nodes, and values are lists of children nodes.
- `an_Y_graph_TF` (bool): True if a returned graph is required to be an ancestor of a list Y.
- `Y` (list): A list of non-overlapping variables.
- `node_positions` (dict): Position of nodes.

**Returns:**

- `nx.DiGraph`: A directed acyclic graph.

**Usage Example:**

```python
from graph import create_acyclic_graph

graph_dict = {
    'T1': ['O1', 'O2'],
    'O1': ['Y'],
    'O2': [],
    'Y': []
}

G = create_acyclic_graph(graph_dict)

```

## Generate All Possible Orders of X

**Function:** `all_possible_orders_X(G, X)`

**Description:** Generate multiple X in different order.

**Parameters:**

- `G` (nx.DiGraph): The directed acyclic graph.
- `X` (list): A set of variables.

**Returns:**

- `X_list` (list of list): A list of X with different orders.

**Usage Example:**

```python
from graph import all_possible_orders_X

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('T1', 'O2'), ('O1', 'Y')])
X = ['T1', 'O1', 'O2']

X_orders = all_possible_orders_X(G, X)
print(X_orders)

```

### Create Subgraphs

**Function:** `subgraphs(G, C)`

**Description:** Create a subgraph of G consisting of nodes in C and unobserved variables that are connected to nodes in C.

**Parameters:**

- `G` (nx.DiGraph): The original causal graph.
- `C` (list): List of observed nodes for the subgraph.

**Returns:**

- `nx.DiGraph`: A subgraph containing nodes in C and connected unobserved variables.

**Usage Example:**

```python
from graph import subgraphs

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('T1', 'O2'), ('O1', 'Y'), ('U1', 'T1')])
C = ['T1', 'O1']

sub_G = subgraphs(G, C)

```

### List All C-Components

**Function:** `list_all_c_components(G)`

**Description:** List all C-components of the graph.

**Parameters:**

- `G` (nx.DiGraph): The graph to analyze.

**Returns:**

- `list`: A list of C-components.

**Usage Example:**

```python
from graph import list_all_c_components

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])

c_components = list_all_c_components(G)
print(c_components)

```

### Find C-Components

**Function:** `find_c_components(G, C)`

**Description:** Find the C-components that contain the nodes in C.

**Parameters:**

- `G` (nx.DiGraph): The graph to analyze.
- `C` (list): The list of nodes to find C-components for.

**Returns:**

- `list`: A list of C-components containing nodes in C.

**Usage Example:**

```python
from graph import find_c_components

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
C = ['T1', 'O1']

c_components = find_c_components(G, C)
print(c_components)

```

### Find Parents

**Function:** `find_parents(G, C)`

**Description:** Find the parents of the nodes in C.

**Parameters:**

- `G` (nx.DiGraph): The graph to analyze.
- `C` (list): The list of nodes to find parents for.

**Returns:**

- `list`: A list of parent nodes.

**Usage Example:**

```python
from graph import find_parents

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
C = ['O1']

parents = find_parents(G, C)
print(parents

)

```

### Find Descendants

**Function:** `find_descendant(G, nodes)`

**Description:** Find all descendants of the given nodes.

**Parameters:**

- `G` (nx.DiGraph): The graph to analyze.
- `nodes` (list): The list of nodes to find descendants for.

**Returns:**

- `list`: A list of descendant nodes.

**Usage Example:**

```python
from graph import find_descendant

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
nodes = ['T1']

descendants = find_descendant(G, nodes)
print(descendants)

```

### Find Ancestors

**Function:** `find_ancestor(G, nodes)`

**Description:** Find all ancestors of the given nodes.

**Parameters:**

- `G` (nx.DiGraph): The graph to analyze.
- `nodes` (list): The list of nodes to find ancestors for.

**Returns:**

- `list`: A list of ancestor nodes.

**Usage Example:**

```python
from graph import find_ancestor

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
nodes = ['O1']

ancestors = find_ancestor(G, nodes)
print(ancestors)

```

### Subgraph of Ancestors of Y

**Function:** `subgraph_ancestor_Y(G, Y)`

**Description:** Create a subgraph of G consisting of the ancestors of nodes in Y.

**Parameters:**

- `G` (nx.DiGraph): The original graph.
- `Y` (list): List of nodes to find ancestors for.

**Returns:**

- `nx.DiGraph`: A subgraph containing the ancestors of nodes in Y.

**Usage Example:**

```python
from graph import subgraph_ancestor_Y

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
Y = ['Y']

sub_G = subgraph_ancestor_Y(G, Y)

```

### Check D-Separation

**Function:** `is_d_separated(G, X, Y, Z)`

**Description:** Check if sets X and Y are d-separated given Z in graph G.

**Parameters:**

- `G` (nx.DiGraph): The graph to analyze.
- `X` (list): List of nodes in set X.
- `Y` (list): List of nodes in set Y.
- `Z` (list): List of conditioning nodes in set Z.

**Returns:**

- `bool`: True if X and Y are d-separated given Z, False otherwise.

**Usage Example:**

```python
from graph import is_d_separated

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
X = ['T1']
Y = ['Y']
Z = ['O1']

d_separated = is_d_separated(G, X, Y, Z)
print(d_separated)

```

### Cut Incoming Edges

**Function:** `G_cut_incoming_edges(G, X)`

**Description:** Remove all incoming edges to nodes in X in graph G.

**Parameters:**

- `G` (nx.DiGraph): The graph to modify.
- `X` (list): List of nodes to remove incoming edges from.

**Returns:**

- `nx.DiGraph`: The modified graph.

**Usage Example:**

```python
from graph import G_cut_incoming_edges

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
X = ['O1']

modified_G = G_cut_incoming_edges(G, X)

```

### Cut Outgoing Edges

**Function:** `G_cut_outgoing_edges(G, X)`

**Description:** Remove all outgoing edges from nodes in X in graph G.

**Parameters:**

- `G` (nx.DiGraph): The graph to modify.
- `X` (list): List of nodes to remove outgoing edges from.

**Returns:**

- `nx.DiGraph`: The modified graph.

**Usage Example:**

```python
from graph import G_cut_outgoing_edges

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
X = ['T1']

modified_G = G_cut_outgoing_edges(G, X)

```

### Generate Random Graph

**Function:** `generate_random_graph(num_observables, num_unobservables, num_treatments, num_outcomes)`

**Description:** Generate a random causal graph with specified numbers of observables, unobservables, treatments, and outcomes.

**Parameters:**

- `num_observables` (int): Number of observable variables.
- `num_unobservables` (int): Number of unobservable variables.
- `num_treatments` (int): Number of treatment variables.
- `num_outcomes` (int): Number of outcome variables.

**Returns:**

- `nx.DiGraph`: The generated random causal graph.

**Usage Example:**

```python
from graph import generate_random_graph

random_G = generate_random_graph(5, 2, 1, 1)

```

### Check Inducing Paths

**Function:** `check_inducing_paths(G, nodes1, nodes2, S, L)`

**Description:** Check for inducing paths between two sets of nodes given sets S and L.

**Parameters:**

- `G` (nx.DiGraph): The graph to analyze.
- `nodes1` (list): The first set of nodes.
- `nodes2` (list): The second set of nodes.
- `S` (list): The set of nodes S.
- `L` (list): The set of nodes L.

**Returns:**

- `bool`: True if there are inducing paths, False otherwise.

**Usage Example:**

```python
from graph import check_inducing_paths

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
nodes1 = ['T1']
nodes2 = ['Y']
S = ['O1']
L = ['U1']

inducing_paths = check_inducing_paths(G, nodes1, nodes2, S, L)
print(inducing_paths)

```

### Check Inducing Paths with Unmeasured Variables

**Function:** `is_inducing_path_with_unmeasured(G, nodes1, nodes2)`

**Description:** Check if there are inducing paths with unmeasured variables between two sets of nodes.

**Parameters:**

- `G` (nx.DiGraph): The graph to analyze.
- `nodes1` (list): The first set of nodes.
- `nodes2` (list): The second set of nodes.

**Returns:**

- `bool`: True if there are inducing paths with unmeasured variables, False otherwise.

**Usage Example:**

```python
from graph import is_inducing_path_with_unmeasured

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
nodes1 = ['T1']
nodes2 = ['Y']

inducing_paths = is_inducing_path_with_unmeasured(G, nodes1, nodes2)
print(inducing_paths)

```

### Check if Node is Unmeasured

**Function:** `is_unmeasured(node)`

**Description:** Check if a node is unmeasured.

**Parameters:**

- `node` (str): The node to check.

**Returns:**

- `bool`: True if the node is unmeasured, False otherwise.

**Usage Example:**

```python
from graph import is_unmeasured

node = 'U1'
unmeasured = is_unmeasured(node)
print(unmeasured)

```

### Find Variables with No Inducing Paths

**Function:** `find_variables_no_inducing_path(G, nodes)`

**Description:** Find variables that have no inducing paths to the given nodes.

**Parameters:**

- `G` (nx.DiGraph): The graph to analyze.
- `nodes` (list): The list of nodes to check for inducing paths.

**Returns:**

- `list`: A list of variables with no inducing paths.

**Usage Example:**

```python
from graph import find_variables_no_inducing_path

G = nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
nodes = ['Y']

no_inducing_path_vars = find_variables_no_inducing_path(G, nodes)
print(no_inducing_path_vars)

```

### Reachable Set

**Function:** `reacheable_set(G, X, A, Z)`

**Description:** Find the set of nodes reachable from X given sets A and Z.

**Parameters:**

- `G` (nx.DiGraph): The graph to analyze.
- `X` (list): The set of nodes X.
- `A` (list): The set of nodes A.
- `Z` (list): The set of nodes Z.

**Returns:**

- `set`: A set of reachable nodes.

**Usage Example:**

```python
from graph import reacheable_set

G =

 nx.DiGraph()
G.add_edges_from([('T1', 'O1'), ('O1', 'Y'), ('U1', 'T1')])
X = ['T1']
A = ['O1']
Z = ['Y']

reachable_nodes = reacheable_set(G, X, A, Z)
print(reachable_nodes)

```

### Convert Graph Dictionary to Fusion Graph

**Function:** `graph_dict_to_fusion_graph(graph_dict)`

**Description:** Convert a graph dictionary to a fusion graph.

**Parameters:**

- `graph_dict` (dict): The graph dictionary to convert.

**Returns:**

- `nx.DiGraph`: The fusion graph.

**Usage Example:**

```python
from graph import graph_dict_to_fusion_graph

graph_dict = {
    'T1': ['O1', 'O2'],
    'O1': ['Y'],
    'O2': [],
    'Y': []
}

fusion_graph = graph_dict_to_fusion_graph(graph_dict)

```

### Example Usage

```python
import networkx as nx
from graph import visualize, write_joint_distribution, find_topological_order, create_acyclic_graph, all_possible_orders_X, subgraphs, list_all_c_components, find_c_components, find_parents, find_descendant, find_ancestor, subgraph_ancestor_Y, is_d_separated, G_cut_incoming_edges, G_cut_outgoing_edges, generate_random_graph, check_inducing_paths, is_inducing_path_with_unmeasured, is_unmeasured, find_variables_no_inducing_path, reacheable_set, graph_dict_to_fusion_graph

# Define a graph dictionary
graph_dict = {
    'T1': ['O1', 'O2'],
    'O1': ['Y'],
    'O2': [],
    'Y': []
}

# Create acyclic graph
G = create_acyclic_graph(graph_dict)

# Get topological order
topological_order = find_topological_order(G)
print(topological_order)  # Output: ['T1', 'O1', 'O2', 'Y']

# Write joint distribution
joint_distribution = write_joint_distribution(topological_order)
print(joint_distribution)  # Output: P(T1 , O1 , O2 , Y)

# Visualize the graph
visualize(G)

# Generate all possible orders of X
X = ['T1', 'O1', 'O2']
X_orders = all_possible_orders_X(G, X)
print(X_orders)

# Create a subgraph
C = ['T1', 'O1']
sub_G = subgraphs(G, C)

# List all C-components
c_components = list_all_c_components(G)
print(c_components)

# Find C-components containing specific nodes
c_components = find_c_components(G, C)
print(c_components)

# Find parents of specific nodes
parents = find_parents(G, C)
print(parents)

# Find descendants of specific nodes
descendants = find_descendant(G, C)
print(descendants)

# Find ancestors of specific nodes
ancestors = find_ancestor(G, C)
print(ancestors)

# Create a subgraph of ancestors of Y
sub_G = subgraph_ancestor_Y(G, ['Y'])

# Check d-separation
d_separated = is_d_separated(G, ['T1'], ['Y'], ['O1'])
print(d_separated)

# Cut incoming edges
modified_G = G_cut_incoming_edges(G, ['O1'])

# Cut outgoing edges
modified_G = G_cut_outgoing_edges(G, ['T1'])

# Generate a random graph
random_G = generate_random_graph(5, 2, 1, 1)

# Check inducing paths
inducing_paths = check_inducing_paths(G, ['T1'], ['Y'], ['O1'], ['U1'])
print(inducing_paths)

# Check inducing paths with unmeasured variables
inducing_paths = is_inducing_path_with_unmeasured(G, ['T1'], ['Y'])
print(inducing_paths)

# Check if a node is unmeasured
unmeasured = is_unmeasured('U1')
print(unmeasured)

# Find variables with no inducing paths
no_inducing_path_vars = find_variables_no_inducing_path(G, ['Y'])
print(no_inducing_path_vars)

# Find reachable set
reachable_nodes = reacheable_set(G, ['T1'], ['O1'], ['Y'])
print(reachable_nodes)

# Convert graph dictionary to fusion graph
fusion_graph = graph_dict_to_fusion_graph(graph_dict)

```

---

This comprehensive documentation covers the functions in `graph.py` and provides detailed usage examples, similar to the provided example documentation for "[SCM.py](http://scm.py/)". If there are additional details or specific formatting requirements, please let me know!