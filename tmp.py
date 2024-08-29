import itertools
import numpy as np

def compute_initial_probabilities(S0, pa_s0):
	"""
	Computes Q[S0](s0) for all possible s0 given pa_s0.
	For simplicity, assume a simple probability model P(s0 | pa_s0).
	"""
	# Generate all possible realizations of S0 (binary variables)
	domain_S0 = list(itertools.product([0, 1], repeat=len(S0)))
	
	# Example: Uniform probability for demonstration
	Q_S0 = {s0: np.random.rand() for s0 in domain_S0}
	
	# Normalize to make it a valid probability distribution
	total = sum(Q_S0.values())
	Q_S0 = {s0: q / total for s0, q in Q_S0.items()}
	
	return Q_S0

def get_variable_indices(Si, S_prev):
	"""
	Get the indices of variables in Si within S_prev.
	"""
	return [S_prev.index(var) for var in Si]


def compute_conditional_probability(Q_S_prev, Si, S_prev):
	"""
	Computes Q[Si](si) using Q[S_prev](s_prev), where Si is an arbitrary subset of S_prev.
	"""
	Q_Si = {}
	
	# Generate all possible realizations of Si
	domain_Si = list(itertools.product([0, 1], repeat=len(Si)))

	# Get indices of Si's variables in S_prev
	indices_Si = get_variable_indices(Si, S_prev)
	
	for si in domain_Si:
		prob = 1.0
		for j, Vj in enumerate(Si):
			# Summation over the variables in S_prev that are not in Si[j:]
			sum_numerator = 0.0
			sum_denominator = 0.0
			
			for s_prev in Q_S_prev.keys():
				# Projection of s_prev onto the first j+1 variables of Si
				s_prev_proj_upto_j = tuple(s_prev[idx] for idx in indices_Si[:j+1]) # s_prev[si_0, si_1, ..., si_j]
				# The corresponding projection of si
				si_proj_upto_j = si[:j+1] # si_0, ..., si_j

				# Projection of s_prev onto the first j variables of Si
				s_prev_proj_upto_j_minus_1 = tuple(s_prev[idx] for idx in indices_Si[:j]) # s_prev[si_0, si_1, ..., si_j]
				# The corresponding projection of si
				si_proj_upto_j_minus_1 = si[:j] # si_0, ..., si_j

				# Accumulate the sum for the numerator
				if s_prev_proj_upto_j == si_proj_upto_j:
					sum_numerator += Q_S_prev[s_prev] # \sum_{s_prev}Q[S_prev]()
				
				# Accumulate the sum for the denominator
				if s_prev_proj_upto_j_minus_1 == si_proj_upto_j_minus_1:
					sum_denominator += Q_S_prev[s_prev]
			
			if sum_denominator != 0:
				prob *= (sum_numerator / sum_denominator)
		
		Q_Si[si] = prob
	
	return Q_Si

if __name__ == "__main__":
	# Example sets of variables
	S0 = ['V1', 'V2', 'V3']
	S1 = ['V1', 'V2']
	S2 = ['V1']
	
	# Known vector for P(s0 | pa_s0) - not used in this example
	pa_s0 = [0.5, 0.5, 0.5]  # Placeholder
	
	# Step 1: Compute Q[S0](s0)
	Q_S0 = compute_initial_probabilities(S0, pa_s0)
	print("Q[S0]:")
	for k, v in Q_S0.items():
		print(f"{k}: {v:.4f}")
	
	# Step 2: Compute Q[S1](s1) from Q[S0](s0)
	Q_S1 = compute_conditional_probability(Q_S0, S1, S0)
	print("\nQ[S1]:")
	for k, v in Q_S1.items():
		print(f"{k}: {v:.4f}")
	
	# Step 3: Compute Q[S2](s2) from Q[S1](s1)
	Q_S2 = compute_conditional_probability(Q_S1, S2, S1)
	print("\nQ[S2]:")
	for k, v in Q_S2.items():
		print(f"{k}: {v:.4f}")
