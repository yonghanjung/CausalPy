import networkx as nx
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import graph
from scipy.special import expit, logit

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

	def generate_random_scm_test(self, num_observables, num_unobservables, num_treatments, num_outcomes, sparcity_constant = 0.5, seed = None, discrete = False):
		'''
		Generate a random acyclic graph with specified numbers of observables, unobservables, treatments, and outcomes.

		Parameters:
		num_observables (int): Number of observable nodes.
		num_unobservables (int): Number of unobservable nodes.
		num_treatments (int): Number of treatment variables.
		num_outcomes (int): Number of outcome variables.

		Returns:
		tuple: Graph dictionary, node positions, lists X and Y.
		'''

		# Create observable nodes
		treatments = [f'X{i+1}' for i in range(num_treatments)]
		outcomes = [f'Y{i+1}' for i in range(num_outcomes)]
		other_observables = [f'V{i+1}' for i in range(num_observables - num_treatments - num_outcomes)]
		all_observables = treatments + outcomes + other_observables
		is_acyclic = False

		while not is_acyclic:
			self.graph.clear()
			self.equations.clear()
			self.noise_distributions.clear()

			# Add unobserved variables
			for i in range(num_unobservables):
				self.add_unobserved_variable(f'U{i+1}', stats.norm(0, 1))

			# Assign two childs to each unobserved variables 
			# unobservable_edges = set()
			for i in range(num_unobservables):
				obs_pair = random.sample(all_observables, 2)
				u_var = f'U_{"_".join(obs_pair)}'
				self.add_unobserved_variable(u_var, stats.norm(0, 1))
				for child in obs_pair:
					self.graph.add_edge(u_var, child)

			additional_edges = [(a, b) for a in all_observables for b in all_observables if a != b]
			random.shuffle(additional_edges)
			additional_edges = random.sample(additional_edges, round(len(additional_edges)*sparcity_constant))

			for edge in additional_edges:
				self.graph.add_edge(*edge)
				if not nx.is_directed_acyclic_graph(self.graph):
					self.graph.remove_edge(*edge)

			if nx.is_directed_acyclic_graph(self.graph):
				if set(outcomes).issubset(set(self.graph.nodes)):
					is_acyclic = False
					for var_name in all_observables:
						parents = graph.find_parents(self.graph, [var_name])
						
						if discrete == False:
							# equation_type = 'binary' if var_name.startswith('X') or var_name.startswith('Y') else 'linear'
							if var_name.startswith('X'):
								equation_type = 'binary' 
								equation = self.create_binary_equation(parents)
							elif var_name.startswith('Y'):
								equation_type = 'linear_bound' 
								equation = self.create_random_linear_bounded_equation(parents)
							else:
								equation_type = 'linear'
								equation = self.create_random_linear_equation(parents)
						else:
							if var_name.startswith('X'):
								equation_type = 'binary' 
								equation = self.create_binary_equation(parents)
							elif var_name.startswith('Y'):
								equation_type = 'binary' 
								equation = self.create_binary_equation(parents)
							else:
								equation_type = 'binary'
								equation = self.create_binary_equation(parents)
						
						# equation = self.create_binary_equation(parents) if equation_type == 'binary' else self.create_random_linear_equation(parents)
						noise_dist = stats.bernoulli(0.5) if equation_type == 'binary' else stats.norm(0, 0.1)
						self.add_observed_variable(var_name, equation, parents, noise_dist)
					break 
				else:
					continue


	# def generate_random_scm(self, num_observed, num_unobserved, num_treatments, num_outcomes, seed = None):
	# 	"""
	# 	Generate a random Structural Causal Model (SCM).

	# 	Parameters:
	# 	num_unobserved (int): Number of unobserved variables.
	# 	num_observed (int): Number of observed variables.
	# 	num_treatments (int): Number of treatment variables.
	# 	num_outcomes (int): Number of outcome variables.
	# 	"""
	# 	if seed is not None: 
	# 		random.seed(int(seed))
	# 		np.random.seed(seed)

	# 	# Initialize observed variable names based on their types (treatment, outcome, or other)
	# 	observed_vars = []
	# 	for i in range(num_treatments):
	# 		var_type = 'X'
	# 		observed_vars.append(f'{var_type}{i+1}')

	# 	for i in range(num_outcomes):
	# 		var_type = 'Y'
	# 		observed_vars.append(f'{var_type}{i+1}')

	# 	for i in range(num_observed - num_treatments - num_outcomes ):
	# 		var_type = 'V'
	# 		observed_vars.append(f'{var_type}{i+1}')

	# 	# Attempt to create an acyclic graph
	# 	while True:
	# 		self.graph.clear()
	# 		self.equations.clear()
	# 		self.noise_distributions.clear()

	# 		# Add unobserved variables
	# 		for i in range(num_unobserved):
	# 			self.add_unobserved_variable(f'U{i+1}', stats.norm(0, 1))

	# 		# Randomly connect observed variables ensuring no isolated variables
	# 		if not self.connect_observed_variables(observed_vars, num_observed):
	# 			continue  # Retry if graph is not valid

	# 		# Assign two observed variables to each unobserved variable
	# 		if not self.assign_unobserved_parents(observed_vars):
	# 			continue  # Retry if graph is not valid

	# 		# Check if the resultant graph is acyclic
	# 		if nx.is_directed_acyclic_graph(self.graph):
	# 			break  # Valid graph generated


	def generate_random_graph(num_observables, num_unobservables, num_treatments, num_outcomes, sparcity_constant = 0.25, seednum = 123):
		'''
		Generate a random acyclic graph with specified numbers of observables, unobservables, treatments, and outcomes.

		Parameters:
		num_observables (int): Number of observable nodes.
		num_unobservables (int): Number of unobservable nodes.
		num_treatments (int): Number of treatment variables.
		num_outcomes (int): Number of outcome variables.

		Returns:
		tuple: Graph dictionary, node positions, lists X and Y.
		'''

		np.random.seed(seednum)
		random.seed(seednum)

		# Create observable nodes
		treatments = [f'X{i+1}' for i in range(num_treatments)]
		outcomes = [f'Y{i+1}' for i in range(num_outcomes)]
		other_observables = [f'V{i+1}' for i in range(num_observables - num_treatments - num_outcomes)]

		all_observables = treatments + outcomes + other_observables
		is_acyclic = False

		while not is_acyclic:
			G = nx.DiGraph()

			# Add unobservable edges
			unobservable_edges = set()
			for i in range(num_unobservables):
				obs_pair = random.sample(all_observables, 2)
				unobservable = f'U_{"_".join(obs_pair)}'
				for child in obs_pair:
					unobservable_edges.add((unobservable, child))

			G.add_edges_from(unobservable_edges)

			# Optional: Add additional edges between observables
			additional_edges = [(a, b) for a in all_observables for b in all_observables if a != b]
			random.shuffle(additional_edges)
			additional_edges = random.sample(additional_edges, round(len(additional_edges)*sparcity_constant))
			for edge in additional_edges:
				G.add_edge(*edge)
				if not nx.is_directed_acyclic_graph(G):
					G.remove_edge(*edge)

			if nx.is_directed_acyclic_graph(G):
				if set(outcomes).issubset(set(G.nodes)):
					break 
				else:
					continue


		# Generate node positions for visualization
		node_positions = {node: (random.uniform(0, 100), random.uniform(0, 100)) for node in G.nodes()}

		# Convert graph to dictionary format
		graph_dict = {node: list(G.successors(node)) for node in G.nodes()}

		return [graph_dict, node_positions, treatments, outcomes]

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
		return pd.DataFrame(self.sample_dict)

	def visualize(self):
		""" Visualize the causal graph with colored nodes for treatments and outcomes. """
		color_map = []
		for node in self.graph:
			if node.startswith('T'):
				color_map.append('blue')  # Treatment nodes colored blue
			elif node.startswith('O'):
				color_map.append('red')  # Outcome nodes colored red
			elif node.startswith('U'):
				color_map.append('gray')  # Unobserved nodes colored gray
			else:
				color_map.append('lightblue')  # Other nodes

		pos = nx.spring_layout(self.graph)
		nx.draw(self.graph, pos, with_labels=True, node_color=color_map, font_weight='bold', arrows=True)
		plt.show()

	

if __name__ == "__main__":
	'''
	Example usage for generating a random SCM
	'''
	scm = StructuralCausalModel()
	scm.generate_random_scm(num_unobserved=1, num_observed=4, num_treatments=1, num_outcomes=1)
	sample_data = scm.generate_samples(100)
	print(sample_data)
	scm.visualize()

	'''
	Example for creating the data geneating process
	'''
	def equation_Z1(U1, num_sample, noise=0):
		return stats.norm(0, 1).rvs(size = num_sample) + U1

	def equation_Z2(U1, num_sample, noise = 0):
		return stats.norm(0, 1).rvs(size = num_sample) + U1

	def equation_X(Z1, Z2, num_sample, noise = 0):
		treatment_prob = 1 / (1 + np.exp(- (0.5 * Z1 + 0.3 * Z2)))
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

	

