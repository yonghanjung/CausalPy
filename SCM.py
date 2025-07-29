import networkx as nx
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import graph
from scipy.special import expit, logit
from pyvis.network import Network


class StructuralCausalModel:
	def __init__(self):
		self.graph = nx.DiGraph()
		self.equations = {}
		self.unobserved_variables = {}
		self.noise_distributions = {}
		self.sample_dict = {}

	def add_unobserved_variable(self, variable, distribution):
		''' Add an unobserved variable with a specified distribution.'''
		self.unobserved_variables[variable] = distribution

	def add_observed_variable(self, variable, equation, parents, noise_distribution, num_samples = None):
		''' Add an observed variable with a specified equation and noise distribution. '''
		self.graph.add_node(variable)
		self.equations[variable] = equation
		self.noise_distributions[variable] = noise_distribution
		for parent in parents:
			self.graph.add_edge(parent, variable)

	def compute(self, variable, num_samples):
		''' Compute the value of an observed variable. '''
		# Check if samples for the variable are already generated. 
		if variable in self.sample_dict:
			return self.sample_dict[variable]

		# If a variable is unobserved, then simply generate unobserved samples. 
		if variable in self.unobserved_variables:
			self.sample_dict[variable] = self.unobserved_variables[variable].rvs(size=num_samples)
			return self.sample_dict[variable]

		# If a variable is observed but there is no equation, then ERROR! 
		if variable not in self.equations:
			raise ValueError(f"No equation defined for observed variable '{variable}'.")

		# For the case where f_{Vi}(PAi, Ui) defined (i.e., PAi, Ui are not empty)
		args = {} # Collection of samples of (PAi, Ui)
		for parent in self.graph.predecessors(variable): # parent in (PAi, Ui)
			if parent in self.sample_dict:
				args[parent] = self.sample_dict[parent]
			else:
				args[parent] = self.compute(parent, num_samples)

		if variable not in self.unobserved_variables:
			noise = self.noise_distributions[variable].rvs(size=num_samples)
			self.sample_dict[variable] = self.equations[variable](**args, noise=noise, num_sample = num_samples)
			# output = self.equations[variable](**args, noise=noise, num_sample = num_samples)

			# # Check if the output is multivariate
			# if isinstance(output, np.ndarray) and len(output.shape) == 2:
			# 	# Assuming the output is of shape (num_samples, num_dimensions)
			# 	for i in range(output.shape[1]):
			# 		self.sample_dict[f'{variable}{i+1}'] = output[:, i]
			# else:
			# 	self.sample_dict[variable] = output
		
		return self.sample_dict[variable]


	def create_random_linear_equation(self, parents):
		""" Create a random linear equation for an observed variable. """
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

	def create_random_linear_bounded_equation(self, parents):
		""" Create a random linear equation for an observed variable. """
		coefficients = {var: random.uniform(-1, 1) for var in parents}
		intercept = random.uniform(-1, 1)

		def linear_equation(**args):
			num_samples = args.pop('num_sample')
			noise = args.pop('noise')
			result = np.zeros(num_samples)
			for var, coef in coefficients.items():
				result += coef * args[var]
			return expit(result + intercept + noise)

		return linear_equation

	def create_binary_equation(self, parents):
		""" Create a binary equation for treatment and outcome variables. """
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
	
	def build_from_graph(self, graph_dict, discrete=False):
		"""
		Builds the SCM equations and internal graph from a pre-defined graph structure.
		This is the proper way to create an SCM from a found graph.
		"""
		self.graph.clear()
		self.equations.clear()
		self.noise_distributions.clear()
		self.unobserved_variables.clear()
		self.graph_dict = graph_dict

		# Add all nodes and edges from the dictionary to the networkx graph
		for node, children in graph_dict.items():
			for child in children:
				self.graph.add_edge(node, child)

		all_nodes = list(self.graph.nodes())
		unobserved = [n for n in all_nodes if n.startswith('U')]
		observables = [n for n in all_nodes if not n.startswith('U')]

		# Add unobserved variables with standard distributions
		for u_var in unobserved:
			self.add_unobserved_variable(u_var, stats.norm(0, 1))

		# For each observable variable, assign a random functional form
		for var_name in observables:
			parents = list(self.graph.predecessors(var_name))
			
			# Determine if the variable should be binary or continuous
			is_binary = discrete or var_name.startswith('X') or var_name.startswith('Y')

			if is_binary:
				equation = self.create_binary_equation(parents)
				noise_dist = stats.norm(0, 1) # Noise for the logit probability
			else: # Continuous
				equation = self.create_random_linear_equation(parents)
				noise_dist = stats.norm(0, 0.1)

			self.add_observed_variable(var_name, equation, parents, noise_dist)

	def generate_random_scm(self, num_observables, num_unobservables, num_treatments,
							num_outcomes, edge_prob=0.5, seed=None, discrete=False):
		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)
			
		# Use the robust external graph generator
		graph_dict, _, _, _ = graph.generate_random_graph(
			num_observables=num_observables,
			num_unobservables=num_unobservables,
			num_treatments=num_treatments,
			num_outcomes=num_outcomes,
			sparcity_constant=edge_prob, # Note the name change
			seednum=seed
		)
		
		# Build the SCM from this generated graph structure
		self.build_from_graph(graph_dict, discrete)

	def assign_unobserved_parents(self, observed_vars):
		"""
		Assign two observed variables to each unobserved variable and ensure acyclicity.
		"""
		for u_var in self.unobserved_variables.keys():
			selected_observed_vars = random.sample(observed_vars, 2)
			for var_name in selected_observed_vars:
				self.graph.add_edge(u_var, var_name)
				if not nx.is_directed_acyclic_graph(self.graph):
					self.graph.remove_edge(u_var, var_name)  # Remove edge if it creates a cycle
					return False
		return True

	def connect_observed_variables(self, observed_vars, num_observed):
		"""
		Randomly connect observed variables and ensure no isolated variables.
		"""
		for var_name in observed_vars:
			possible_parents = [v for v in observed_vars if v != var_name] 
			num_parents = random.randint(0, min(len(possible_parents), int(num_observed / 1.5)))  # Limit number of parents
			parents = random.sample(possible_parents, num_parents)
			equation_type = 'binary' if var_name.startswith('X') or var_name.startswith('Y') else 'linear'
			equation = self.create_binary_equation(parents) if equation_type == 'binary' else self.create_random_linear_equation(parents)
			noise_dist = stats.bernoulli(0.5) if equation_type == 'binary' else stats.norm(0, 0.1)
			self.add_observed_variable(var_name, equation, parents, noise_dist)

		# Check for isolated variables
		for var in observed_vars:
			if self.graph.in_degree(var) == 0 and self.graph.out_degree(var) == 0:
				return False  # Isolated variable found
		return True

	def generate_samples(self, num_samples, seed=None):
		""" Generate n samples from the SCM for observed variables. """
		if seed is not None: 
			random.seed(int(seed))
			np.random.seed(seed)

		self.sample_dict.clear()
		for var in self.equations:  # Iterate only over observed variables
			self.compute(var, num_samples)

		# Prepare a new dictionary to hold the columns for the DataFrame
		processed_dict = {}

		# Iterate through the keys and values of tmp_dict
		for key, value in self.sample_dict.items():
			if isinstance(value, np.ndarray) and len(value.shape) == 2:  # Check if value is multivariate
				# If multivariate, create new columns for each dimension
				for i in range(value.shape[1]):
					processed_dict[f'{key}{i+1}'] = value[:, i]
			else:
				# If univariate, add it directly
				processed_dict[key] = value

		return pd.DataFrame(processed_dict)

	def generate_observational_samples(self, num_samples, seed=None):
		"""
		Generates samples and returns only the observed variables.

		This is a convenience wrapper around `generate_samples` that filters
		out all unobserved exogenous variables (columns starting with 'U').

		Parameters:
			n_samples (int): The number of samples to generate.

		Returns:
			pandas.DataFrame: A DataFrame containing only the observed variables.
		"""
		full_data = self.generate_samples(num_samples)

		# Filter out columns that start with 'U'
		observational_cols = [
			col for col in full_data.columns if not col.startswith('U')
		]

		return full_data[observational_cols]

	def get_adjacency_matrix(self):
		"""
		Returns the adjacency matrix of the causal graph as a pandas DataFrame.

		In the matrix, the value at row `i` and column `j` is 1 if there is a
		directed edge from node `i` to node `j`, and 0 otherwise.
		"""
		# Get a sorted list of nodes to ensure consistent matrix order
		node_list = sorted(list(self.graph.nodes()))
		# Get the adjacency matrix as a SciPy sparse matrix
		adj_matrix_sparse = nx.adjacency_matrix(self.graph, nodelist=node_list)
		# Convert to a dense pandas DataFrame for easy viewing
		df = pd.DataFrame(adj_matrix_sparse.toarray(), index=node_list, columns=node_list)
		return df

if __name__ == "__main__":
	# --- Test Case for StructuralCausalModel Class ---
	
	# 1. Define parameters for a random SCM and generate it
	print("--- 1. Generating a random SCM ---")
	scm_test = StructuralCausalModel()
	scm_test.generate_random_scm(
		num_observables=5,
		num_unobservables=2,
		num_treatments=1,
		num_outcomes=1,
		seed=42,
		discrete=True  # Make X and Y binary
	)
	print("SCM created successfully.")
	
	# 2. Generate sample data from the model
	print("\n--- 2. Generating sample data ---")
	try:
		sample_data = scm_test.generate_samples(100, seed=42)
		print("Generated DataFrame head:")
		print(sample_data.head())
	except Exception as e:
		print(f"An error occurred during sample generation: {e}")

	# 3. Visualize the graph structure
	print("\n--- 3. Visualizing the graph ---")
	try:
		graph.visualize_interactive(scm_test.graph)
	except Exception as e:
		print(f"An error occurred during visualization: {e}")

	

