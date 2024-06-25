This document will guide you through the process of understanding and using a Python implementation of a Structural Causal Model (SCM). The SCM allows users to generate random causal graphs and corresponding datasets or define their own equations and generate samples from them. This document explains the code, its components, and how to use it.

# Table of Contents

1. [Introduction](#Introduction)
2. [Setup](#Setup)
3. [Modules](#Modules)
    1. [Fundamental Modules](#Fundamental-Modules)
        1. [Initialization](#Initialization) 
        2. [Adding variables](#Adding-Variables)
        3. [Generate samples from observables](#Generate-Samples-from-Observables)
    2. [Create random SCM](#Generating-a-Random-SCM)
        1. [Creating random equations](https://www.notion.so/DevDoc-SCM-py-284541c25c4a45bbaff38759eebb51ff?pvs=21) 
        2. [Assigning unobserved parents](https://www.notion.so/DevDoc-SCM-py-284541c25c4a45bbaff38759eebb51ff?pvs=21)
    3. [Visualization](https://www.notion.so/DevDoc-SCM-py-284541c25c4a45bbaff38759eebb51ff?pvs=21)
    4. [Generate samples](https://www.notion.so/DevDoc-SCM-py-284541c25c4a45bbaff38759eebb51ff?pvs=21)
4. [Example usage](https://www.notion.so/DevDoc-SCM-py-284541c25c4a45bbaff38759eebb51ff?pvs=21)
    1. [Generate random SCM](https://www.notion.so/DevDoc-SCM-py-284541c25c4a45bbaff38759eebb51ff?pvs=21)
    2. [Create a custom SCM](https://www.notion.so/DevDoc-SCM-py-284541c25c4a45bbaff38759eebb51ff?pvs=21)

# Introduction

A Structural Causal Model (SCM) is a framework for modeling causal relationships between variables. It consists of a directed acyclic graph (DAG) where nodes represent variables, and edges represent causal effects. This tutorial demonstrates how to implement an SCM using Python, generate random SCMs, and define custom causal relationships.

# Setup

First, ensure you have the necessary libraries installed:

```bash
pip install networkx scipy matplotlib numpy pandas
```

Then, import the required libraries and define the `StructuralCausalModel` class.

```python
import networkx as nx
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
```

# Modules

## Fundamental Modules

The `StructuralCausalModel` class contains methods for defining and working with causal models.

### Initialization

The `__init__` method initializes an empty graph and dictionaries to store equations, noise distributions, and samples.

```python
class StructuralCausalModel:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.equations = {}
        self.unobserved_variables = {}
        self.noise_distributions = {}
        self.sample_dict = {}

```

### Adding Variables

The `add_unobserved_variable` and `add_observed_variable` methods add unobserved and observed variables to the model.

```python
def add_unobserved_variable(self, variable, distribution):
    self.unobserved_variables[variable] = distribution

def add_observed_variable(self, variable, equation, parents, noise_distribution):
    self.graph.add_node(variable)
    self.equations[variable] = equation
    self.noise_distributions[variable] = noise_distribution
    for parent in parents:
        self.graph.add_edge(parent, variable)

```

### Generate Samples from Observables

The `compute` method generates samples for observed variables based on the structural equation taking their parents and noise as arguments. 

```python
def compute(self, variable, num_samples):
    if variable in self.sample_dict:
        return self.sample_dict[variable]

    if variable in self.unobserved_variables:
        self.sample_dict[variable] = self.unobserved_variables[variable].rvs(size=num_samples)
        return self.sample_dict[variable]

    if variable not in self.equations:
        raise ValueError(f"No equation defined for observed variable '{variable}'.")

    args = {}
    for parent in self.graph.predecessors(variable):
        if parent in self.sample_dict:
            args[parent] = self.sample_dict[parent]
        else:
            args[parent] = self.compute(parent, num_samples)

    noise = self.noise_distributions[variable].rvs(size=num_samples)
    self.sample_dict[variable] = self.equations[variable](**args, noise=noise, num_sample=num_samples)

    return self.sample_dict[variable]
```

## Generating a Random SCM

The `generate_random_scm` method generates a random SCM with specified numbers of unobserved, observed, treatment, and outcome variables. It asserts that a created SCM doesn’t contain cycles and isolated set of observables. 

```python
def generate_random_scm(self, num_unobserved, num_observed, num_treatments, num_outcomes, seed=None):
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(seed)

    observed_vars = []
    for i in range(num_observed):
        var_type = 'T' if i < num_treatments else 'O' if i >= num_observed - num_outcomes else 'V'
        observed_vars.append(f'{var_type}{i+1}')

    while True:
        self.graph.clear()
        self.equations.clear()
        self.noise_distributions.clear()

        for i in range(num_unobserved):
            self.add_unobserved_variable(f'U{i+1}', stats.norm(0, 1))

        if not self.connect_observed_variables(observed_vars, num_observed):
            continue

        if not self.assign_unobserved_parents(observed_vars):
            continue

        if nx.is_directed_acyclic_graph(self.graph):
            break
```

### Connecting Observed Variables

The `connect_observed_variables` method randomly connects observed variables while ensuring no isolated variables.

```python
def connect_observed_variables(self, observed_vars, num_observed):
    for var_name in observed_vars:
        possible_parents = [v for v in observed_vars if v != var_name]
        num_parents = random.randint(0, min(len(possible_parents), int(num_observed / 2)))
        parents = random.sample(possible_parents, num_parents)
        equation_type = 'binary' if var_name.startswith('T') or var_name.startswith('O') else 'linear'
        equation = self.create_binary_equation(parents) if equation_type == 'binary' else self.create_random_linear_equation(parents)
        noise_dist = stats.bernoulli(0.5) if equation_type == 'binary' else stats.norm(0, 0.1)
        self.add_observed_variable(var_name, equation, parents, noise_dist)

    for var in observed_vars:
        if self.graph.in_degree(var) == 0 and self.graph.out_degree(var) == 0:
            return False
    return True

```

### Creating Random Equations

The `create_random_linear_equation` and `create_binary_equation` methods generate random linear and binary equations for variables.

```python
def create_random_linear_equation(self, parents):
    coefficients = {var: random.uniform(-1, 1) for var in parents}
    intercept = random.uniform(-1, 1)

    def linear_equation(**args):
        num_samples = args.pop('num_sample')
        noise = args.pop('noise')
        result = np.zeros(num_samples)
        for var, coef in coefficients.items():
            result += coef * args[var]
        return result + intercept + noise

    return linear_equation

def create_binary_equation(self, parents):
    coefficients = {var: random.uniform(-1, 1) for var in parents}
    intercept = random.uniform(-1, 1)

    def binary_equation(**args):
        num_samples = args.pop('num_sample')
        noise = args.pop('noise')
        linear_part = np.zeros(num_samples)
        for var, coef in coefficients.items():
            linear_part += coef * args[var]
        linear_part += intercept + noise
        return np.random.binomial(1, 1 / (1 + np.exp(-linear_part)))

    return binary_equation

```

### Assigning Unobserved Parents

The `assign_unobserved_parents` method assigns unobserved variables as parents to observed variables while ensuring the graph remains acyclic.

```python
def assign_unobserved_parents(self, observed_vars):
    for u_var in self.unobserved_variables.keys():
        selected_observed_vars = random.sample(observed_vars, 2)
        for var_name in selected_observed_vars:
            self.graph.add_edge(u_var, var_name)
            if not nx.is_directed_acyclic_graph(self.graph):
                self.graph.remove_edge(u_var, var_name)
                return False
    return True
```

## Visualization

The `visualize` method visualizes the causal graph with different colors for treatments, outcomes, and unobserved variables.

```python
def visualize(self):
    color_map = []
    for node in self.graph:
        if node.startswith('T'):
            color_map.append('blue')
        elif node.startswith('O'):
            color_map.append('red')
        elif node.startswith('U'):
            color_map.append('gray')
        else:
            color_map.append('lightblue')

    pos = nx.spring_layout(self.graph)
    nx.draw(self.graph, pos, with_labels=True, node_color=color_map, font_weight='bold', arrows=True)
    plt.show()

```

## Generating Samples

The `generate_samples` method generates samples from the SCM for observed variables.

```python
def generate_samples(self, num_samples, seed=None):
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(seed)

    self.sample_dict.clear()
    for var in self.equations:
        self.compute(var, num_samples)
    return pd.DataFrame(self.sample_dict)
```

# Example Usage

## Generating a Random SCM

```python
if __name__ == "__main__":
    scm = StructuralCausalModel()
    scm.generate_random_scm(num_unobserved=1, num_observed=4, num_treatments=1, num_outcomes=1)
    sample_data = scm.generate_samples(100)
    print(sample_data)
    scm.visualize()

```

## Creating a Custom SCM

```python
def equation_Z1(U1, num_sample, noise=0):
    return stats.norm(0, 1).rvs(size=num_sample) + U1

def equation_Z2(U1, num_sample, noise=

0):
    return stats.norm(0, 1).rvs(size=num_sample) + U1

def equation_X(Z1, Z2, num_sample, noise=0):
    treatment_prob = 1 / (1 + np.exp(-(0.5 * Z1 + 0.3 * Z2)))
    return np.random.binomial(1, treatment_prob, num_sample)

def equation_Y(Z1, Z2, X, num_sample, noise):
    return 2 * X + 1.5 * Z1 - Z2 + noise

scm = StructuralCausalModel()
scm.add_unobserved_variable('U1', stats.norm(0, 1))
scm.add_observed_variable('Z1', equation_Z1, ['U1'], stats.norm(0, 0.1))
scm.add_observed_variable('Z2', equation_Z2, ['U1'], stats.norm(0, 0.1))
scm.add_observed_variable('X', equation_X, ['Z1', 'Z2'], stats.norm(0, 0.1))
scm.add_observed_variable('Y', equation_Y, ['Z1', 'Z2', 'X'], stats.norm(0, 0.1))
sample_data = scm.generate_samples(100)
scm.visualize()

```