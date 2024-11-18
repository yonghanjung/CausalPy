import random 
import scipy.stats as stats
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

from SCM import StructuralCausalModel  # Ensure generateSCM.py is in the same directory

def inv_logit(vec):
	return 1/(1+np.exp(-vec))

def BD_SCM(seednum = None, d = 10):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_C(noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		first_column = [2**(-abs(j - 0) - 1) for j in range(d)]
		first_row = [2**(-abs(0 - k) - 1) for k in range(d)]
		toeplitz_matrix = toeplitz(first_column, first_row)

		return stats.multivariate_normal.rvs(mean = np.zeros(d), cov = toeplitz_matrix, size=num_samples)

	def equation_X(C, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( np.sum(C,axis=1) + 1 + noise)
		return np.random.binomial(1, prob_X)

	def equation_Y(C, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		Csum = np.sum(C,axis=1)
		prob_Y = inv_logit( 2*(2 * X - 1)*Csum + 0.5 * Csum + (2*X - 1) + noise )
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_observed_variable('C', equation_C, [], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['C'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['C', 'X'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']
	return [scm, X, Y]

def Kang_Schafer(seednum = None):
	# I refer the one in https://arxiv.org/pdf/1704.00211 Section 5.1. 
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_Z1(**kwargs):
		num_samples = kwargs.pop('num_sample')
		return np.random.normal(0, 1, num_samples)

	def equation_Z2(**kwargs):
		num_samples = kwargs.pop('num_sample')
		return np.random.normal(0, 1, num_samples)

	def equation_Z3(**kwargs):
		num_samples = kwargs.pop('num_sample')
		return np.random.normal(0, 1, num_samples)

	def equation_Z4(**kwargs):
		num_samples = kwargs.pop('num_sample')
		return np.random.normal(0, 1, num_samples)

	def equation_X(Z1, Z2, Z3, Z4, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( -Z1 + 0.5 * Z2 - 0.25 * Z3 - 0.1 * Z4  )
		return np.random.binomial(1, prob_X)

	def equation_Y(Z1, Z2, Z3, Z4, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		Y = 210 + X*((27.4 * Z1) + (13.7 * Z2) + (13.7 * Z3) + (13.7 * Z4)) + noise 
		return Y 

	scm = StructuralCausalModel()
	scm.add_observed_variable('Z1', equation_Z1, [], stats.norm(0, 0.1))
	scm.add_observed_variable('Z2', equation_Z2, [], stats.norm(0, 0.1))
	scm.add_observed_variable('Z3', equation_Z3, [], stats.norm(0, 0.1))
	scm.add_observed_variable('Z4', equation_Z4, [], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['Z1', 'Z2', 'Z3', 'Z4'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['Z1', 'Z2', 'Z3', 'Z4','X'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']
	return [scm, X, Y]

def Kang_Schafer_dim(seednum = None, d=4):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_Z(**kwargs):
		num_samples = kwargs.get('num_sample', None)
		return stats.multivariate_normal.rvs(mean = np.zeros(d), cov = np.eye(d), size=num_samples)

	def equation_X(Z, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')

		# coeff = [27.4] + [13.7] * (len(Z_list)-1) 
		coeff = np.array( [(-1)**n * 2**(-n) for n in range(1, d + 1)] )
		X_agg = np.dot(np.array(Z), coeff.T)  # Compute dot product
		prob_X = inv_logit(X_agg)
		return np.random.binomial(1, prob_X)

	def equation_Y(Z, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')

		coeff = np.array( [27.4] + [13.7] * (d - 1) )
		Y_agg = np.dot(np.array(Z), coeff.T)  # Compute dot product
		Y = 210 + X*Y_agg + noise
		return Y

	scm = StructuralCausalModel()
	scm.add_observed_variable('Z', equation_Z, [], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['Z'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['Z','X'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']
	return [scm, X, Y]

def Dukes_Vansteelandt_Farrel(seednum = None, d=200):
	# Inference for treatment effect parameters in potentially misspecified high-dimensional models
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_Z(**kwargs):
		num_samples = kwargs.pop('num_sample')
		first_column = [2**(-abs(j - 0) - 1) for j in range(d)]
		first_row = [2**(-abs(0 - k) - 1) for k in range(d)]
		toeplitz_matrix = toeplitz(first_column, first_row)
		
		return stats.multivariate_normal.rvs(mean = np.zeros(d), cov = toeplitz_matrix, size=num_samples)

	def equation_X(Z, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeffs = [1,-1,1] + [-(i - 2) ** (-2) for i in range(4,d+1)]
		prob_X = inv_logit( np.dot(np.array(coeffs), np.array(Z).T ) )
		return np.random.binomial(1, prob_X)

	def equation_Y(Z,X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeffs = [-1,1,-1] + [(i - 2) ** (-2) for i in range(4,d+1)]
		Y = np.dot(np.array(coeffs), np.array(Z).T ) + 0.3 * X + noise
		return Y
	
	scm = StructuralCausalModel()
	scm.add_observed_variable('Z', equation_Z, [], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['Z'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['Z', 'X'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']
	return [scm, X, Y]

# def mSBD_SCM_timestep(seednum=None, d=4, time_step=3):
# 	if seednum is not None:
# 		random.seed(int(seednum))
# 		np.random.seed(seednum)

# 	scm = StructuralCausalModel()
# 	prev_lists = []

# 	for t in range(1, time_step + 1):
# 		# Define Z equations
# 		def make_equation_Z(prev_lists, noise, **kwargs):
# 			num_samples = kwargs.get('num_sample', None)
# 			starting_nvec = np.zeros(num_samples)
# 			Zt = stats.multivariate_normal.rvs(mean=np.zeros(d), cov=np.eye(d), size=num_samples)

# 			coeffs = [-(i) ** (-2) for i in range(1, d + 1)]
# 			coeff2 = 2

# 			for prev_var in prev_lists:
# 				prev_var = scm.sample_dict[prev_var]
# 				if len(prev_var.shape) == 2:
# 					mean_Z = np.dot(np.array(starting_nvec).T, np.array(prev_var))
# 					mean_Z = (np.max(mean_Z) - mean_Z) / (np.max(mean_Z) - np.min(mean_Z))
# 					Zt += stats.multivariate_normal.rvs(mean=mean_Z, cov=np.eye(d), size=num_samples)
# 				else:
# 					starting_nvec += (coeff2 ** -2) * prev_var
# 					coeff2 += 2

# 			Zt = (np.max(Zt) - Zt) / (np.max(Zt) - np.min(Zt))
# 			return Zt

# 		if t == 1:
# 			parents = []
# 			def equation_Z_wrapper(**kwargs):
# 				return make_equation_Z(parents, **kwargs)
# 			scm.add_observed_variable(f'Z{t}', equation_Z_wrapper, [], stats.norm(0, 0.1))
# 		else:
# 			parents = [f"{var}{i}" for i in range(1, t) for var in ['Z', 'X', 'Y']]
# 			def equation_Z_wrapper(**kwargs):
# 				local_parents = [f"{var}{i}" for i in range(1, t) for var in ['Z', 'X', 'Y']]
# 				return make_equation_Z(local_parents, **kwargs)
# 			scm.add_observed_variable(f'Z{t}', equation_Z_wrapper, parents, stats.norm(0, 0.1))

# 		prev_lists.append(f'Z{t}')

# 		# Define X equations
# 		def make_equation_X(prev_lists, noise, t=t, **kwargs):
# 			num_samples = kwargs.get('num_sample', None)
# 			starting_nvec = np.zeros(num_samples)
# 			coeffs = [-(i) ** (-2) for i in range(1, d + 1)]
# 			coeff2 = 2

# 			for prev_var in prev_lists:
# 				prev_var = scm.sample_dict[prev_var]
# 				if len(prev_var.shape) == 2:
# 					starting_nvec += np.dot(np.array(coeffs), np.array(prev_var).T)
# 				else:
# 					starting_nvec += (coeff2 ** -2) * prev_var
# 					coeff2 += 2

# 			starting_nvec = (np.max(starting_nvec) - starting_nvec) / (np.max(starting_nvec) - np.min(starting_nvec))
# 			prob = inv_logit(starting_nvec)
# 			return np.random.binomial(1, prob)

# 		parents = [f"{var}{i}" for i in range(1, t) for var in ['Z', 'X', 'Y']]
# 		parents.append(f'Z{t}')
# 		def equation_X_wrapper(**kwargs):
# 			local_parents = [f"{var}{i}" for i in range(1, t) for var in ['Z', 'X', 'Y']]
# 			local_parents.append(f'Z{t}')
# 			return make_equation_X(local_parents, **kwargs)
# 		scm.add_observed_variable(f'X{t}', equation_X_wrapper, parents, stats.norm(0, 0.1))

# 		prev_lists.append(f'X{t}')

# 		# Define Y equations
# 		def make_equation_Y(prev_lists, noise, t=t, **kwargs):
# 			num_samples = kwargs.get('num_sample', None)
# 			starting_nvec = np.zeros(num_samples)
# 			coeffs = [-(i) ** (-2) for i in range(1, d + 1)]
# 			coeff2 = 2

# 			for prev_var in prev_lists:
# 				prev_var = scm.sample_dict[prev_var]
# 				if len(prev_var.shape) == 2:
# 					starting_nvec += np.dot(np.array(coeffs), np.array(prev_var).T)
# 				else:
# 					starting_nvec += (coeff2 ** -0.5) * prev_var
# 					coeff2 += 2

# 			starting_nvec = (np.max(starting_nvec) - starting_nvec) / (np.max(starting_nvec) - np.min(starting_nvec))
# 			prob = inv_logit(starting_nvec)
# 			return np.random.binomial(1, prob)

# 		parents = [f"{var}{i}" for i in range(1, t) for var in ['Z', 'X', 'Y']]
# 		parents.append(f'Z{t}')
# 		parents.append(f'X{t}')
# 		def equation_Y_wrapper(**kwargs):
# 			local_parents = [f"{var}{i}" for i in range(1, t) for var in ['Z', 'X', 'Y']]
# 			local_parents.append(f'Z{t}')
# 			local_parents.append(f'X{t}')
# 			return make_equation_Y(local_parents, **kwargs)
# 		scm.add_observed_variable(f'Y{t}', equation_Y_wrapper, parents, stats.norm(0, 0.1))
		
# 		prev_lists.append(f'Y{t}')

# 	# Collect all X and Y variables based on the time step
# 	X = [f'X{t}' for t in range(1, time_step + 1)]
# 	Y = [f'Y{t}' for t in range(1, time_step + 1)]

# 	return [scm, X, Y]


def mSBD_SCM(seednum = None, d=4):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_Z1(noise, **kwargs):
		num_samples = kwargs.get('num_sample', None)
		return stats.multivariate_normal.rvs(mean = np.zeros(d), cov = np.eye(d), size=num_samples)

	def equation_X1(Z1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeffs = [1,-1,1] + [-(i - 2) ** (-2) for i in range(4,d+1)]
		prob = inv_logit( np.dot(np.array(coeffs), np.array(Z1).T ) )
		return np.random.binomial(1, prob)

	def equation_Y1(Z1, X1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeffs = [-1,1,-1] + [(i - 2) ** (-2) for i in range(4,d+1)]
		prob = inv_logit( np.dot(np.array(coeffs), np.array(Z1).T) + 0.3 * X1 + noise )
		return np.random.binomial(1, prob)

	def equation_Z2(Z1, X1, Y1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeffs = [1,-1,1] + [-(i - 2) ** (-2) for i in range(4,d+1)]
		mean_Z2 = np.dot(np.array(X1 + Y1).T, np.array(Z1)) 
		mean_Z2 = (np.max(mean_Z2)-mean_Z2)/(np.max(mean_Z2)-np.min(mean_Z2))
		return stats.multivariate_normal.rvs(mean = mean_Z2, cov = np.eye(d), size=num_samples)

	def equation_X2(Z1, X1, Y1, Z2,  noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeffs_1 = [-1,1,-1] + [(i - 2) ** (-2) for i in range(4,d+1)]
		coeffs_2 = [1,-1,1] + [-(i - 2) ** (-2) for i in range(4,d+1)]
		prob = inv_logit( np.dot(np.array(coeffs_1), np.array(Z1).T) + np.dot(np.array(coeffs_2), np.array(Z2).T) - 0.5 * X1 + 0.3 * Y1 + noise )
		return np.random.binomial(1, prob)

	def equation_Y2(Z1, X1, Y1, Z2, X2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeffs_1 = [1,-1,1] + [-(i - 2) ** (-2) for i in range(4,d+1)]
		coeffs_2 = [-1,1,-1] + [(i - 2) ** (-2) for i in range(4,d+1)]
		prob = inv_logit( np.dot(np.array(coeffs_1), np.array(Z1).T) + np.dot(np.array(coeffs_2), np.array(Z2).T) + 0.3 * X1 + 0.3 * Y1 + 0.3 * X2 + noise )
		return np.random.binomial(1, prob)

	scm = StructuralCausalModel()
	scm.add_observed_variable('Z1', equation_Z1, [], stats.norm(0, 0.1))
	scm.add_observed_variable('X1', equation_X1, ['Z1'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y1', equation_Y1, ['Z1', 'X1'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z2', equation_Z2, ['Z1', 'X1', 'Y1'], stats.norm(0, 0.1))
	scm.add_observed_variable('X2', equation_X2, ['Z1', 'X1', 'Y1', 'Z2'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y2', equation_Y2, ['Z1', 'X1', 'Y1', 'Z2', 'X2'], stats.norm(0, 0.1))	

	X = ['X1', 'X2']
	Y = ['Y1', 'Y2']

	return [scm, X, Y]


def Luedtke_v1(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_Z1(noise, **kwargs):
		num_samples = kwargs.get('num_sample', None)
		return np.random.normal(0,1,size=num_samples)

	def equation_X1(Z1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob = inv_logit( Z1 ) 
		return np.random.binomial(1, prob)

	def equation_Z2(Z1, X1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		return np.random.normal(0,1,size=num_samples)

	def equation_X2(Z1, X1, Z2,  noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob = inv_logit( Z2 + X1 + Z1  )
		return np.random.binomial(1, prob)

	def equation_Z3(Z1, X1, Z2, X2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		Z3 = Z1* X2 + Z2 * X1 + Z2 * X2  + noise 
		return Z3 

	def equation_X3(Z1, X1, Z2, X2, Z3,  noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob = inv_logit( Z1 + X1 + Z2 + X2 + Z3  )
		return np.random.binomial(1, prob)

	def equation_Y(Z1, X1, Z2, X2, Z3, X3, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob = inv_logit( Z2*X3 + X2*Z3 + Z3*X3  )
		return np.random.binomial(1, prob)

	scm = StructuralCausalModel()
	scm.add_observed_variable('Z1', equation_Z1, [], stats.norm(0, 0.1))
	scm.add_observed_variable('X1', equation_X1, ['Z1'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z2', equation_Z2, ['Z1', 'X1'], stats.norm(0, 0.1))
	scm.add_observed_variable('X2', equation_X2, ['Z1', 'X1', 'Z2'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z3', equation_X2, ['Z1', 'X1', 'Z2', 'X2'], stats.norm(0, 0.1))
	scm.add_observed_variable('X3', equation_X2, ['Z1', 'X1', 'Z2', 'X2', 'Z3'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_X2, ['Z1', 'X1', 'Z2', 'X2', 'Z3', 'X3'], stats.norm(0, 0.1))

	X = ['X1', 'X2', 'X3']
	Y = ['Y']

	return [scm, X, Y]


def Luedtke_v2(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_Z1(noise, **kwargs):
		num_samples = kwargs.get('num_sample', None)
		Z = np.abs( np.random.normal(0,1,size=num_samples) )
		return Z 

	def equation_X1(Z1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob = inv_logit( Z1 ) 
		return np.random.binomial(1, prob)

	def equation_Z2(Z1, X1, noise, **kwargs):
		num_samples = kwargs.get('num_sample', None)
		Z = -2*X1 + 0.5 * Z1 +  np.abs( np.random.normal(0,1,size=num_samples) )
		return Z 

	def equation_X2(Z1, X1, Z2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		eq1 = inv_logit( 1.7 - 2*( inv_logit(Z2) > 0.9 ) )
		prob = inv_logit( X1* eq1 )
		return np.random.binomial(1, prob)

	def equation_Y2(Z1, X1, Z2, X2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob = inv_logit( -3 + 0.5*Z1*X2 + 0.5 * X1*Z2 + 0.5 * Z2 * X2 )
		return np.random.binomial(1, prob)

	def equation_Z3(Z1, X1, Z2, X2, Y2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		L3 = np.abs( X1 * Z2 + Z2 * X2 + noise ) 
		Z3 = -2 * 0.5 * Z2 + L3 
		return Z3 

	def equation_X3(Z1, X1, Z2, X2, Y2, Z3, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		eq1 = inv_logit( 1.7 - 2*( inv_logit(Z3) > 0.85 ) )
		prob = inv_logit( X2* eq1 )
		return np.random.binomial(1, prob)

	def equation_Y3(Z1, X1, Z2, X2, Y2, Z3, X3, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob = inv_logit( -3*Y2 + 0.5*Z2*X3 + 0.5 * X2*Z3 + 0.5 * Z3 * X3 )
		return np.random.binomial(1, prob)

	def equation_Z4(Z1, X1, Z2, X2, Y2, Z3, X3, Y3, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		L4 = np.abs( Z2 * X3 + X2 * Z3 + Z3 * X3 + noise ) 
		Z4 = -2 * 0.5 * Z3 + L4 
		return Z4 

	def equation_X4(Z1, X1, Z2, X2, Y2, Z3, X3, Y3, Z4, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		eq1 = inv_logit( 1.7 - 2*( inv_logit(Z4) > 0.8 ) )
		prob = inv_logit( X3* eq1 )
		return np.random.binomial(1, prob)

	def equation_Y4(Z1, X1, Z2, X2, Y2, Z3, X3, Y3, Z4, X4, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob = inv_logit( -3*Y3 + 0.5*Z3*X4 + 0.5 * X3*Z4 + 0.5 * Z4 * X4 )
		return np.random.binomial(1, prob)

	def equation_Z5(Z1, X1, Z2, X2, Y2, Z3, X3, Y3, Z4, X4, Y4, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		L5 = np.abs( Z2 * X4 + X2 * Z4 + Z4 * X4 + noise ) 
		Z5 = -1 + 0.25 * Z4 + 0.5 * L5 - 0.1 * Z4 * L5 + 1.5 * Z4
		return Z5 

	def equation_X5(Z1, X1, Z2, X2, Y2, Z3, X3, Y3, Z4, X4, Y4, Z5, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		eq1 = inv_logit( 2 - 2*( inv_logit(Z5) > 0.8 ) )
		prob = inv_logit( X4* eq1 )
		return np.random.binomial(1, prob)

	def equation_Y5(Z1, X1, Z2, X2, Y2, Z3, X3, Y3, Z4, X4, Y4, Z5, X5, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob = inv_logit( -Y4 + X5 + Z5 * X5 + 0.2 * X4 * Z5 )
		return np.random.binomial(1, prob)

	scm = StructuralCausalModel()
	scm.add_observed_variable('Z1', equation_Z1, [], stats.norm(0, 0.1))
	scm.add_observed_variable('X1', equation_X1, ['Z1'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z2', equation_Z2, ['Z1', 'X1'], stats.norm(0, 0.1))
	scm.add_observed_variable('X2', equation_X2, ['Z1', 'X1', 'Z2'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y2', equation_Y2, ['Z1', 'X1', 'Z2', 'X2'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z3', equation_Z3, ['Z1', 'X1', 'Z2', 'X2', 'Y2'], stats.norm(0, 0.1))
	scm.add_observed_variable('X3', equation_X3, ['Z1', 'X1', 'Z2', 'X2', 'Y2', 'Z3'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y3', equation_Y3, ['Z1', 'X1', 'Z2', 'X2', 'Y2', 'Z3', 'X3'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z4', equation_Z4, ['Z1', 'X1', 'Z2', 'X2', 'Y2', 'Z3', 'X3', 'Y3'], stats.norm(0, 0.1))
	scm.add_observed_variable('X4', equation_X4, ['Z1', 'X1', 'Z2', 'X2', 'Y2', 'Z3', 'X3', 'Y3', 'Z4'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y4', equation_Y4, ['Z1', 'X1', 'Z2', 'X2', 'Y2', 'Z3', 'X3', 'Y3', 'Z4', 'X4'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z5', equation_Z5, ['Z1', 'X1', 'Z2', 'X2', 'Y2', 'Z3', 'X3', 'Y3', 'Z4', 'X4', 'Y4'], stats.norm(0, 0.1))
	scm.add_observed_variable('X5', equation_X5, ['Z1', 'X1', 'Z2', 'X2', 'Y2', 'Z3', 'X3', 'Y3', 'Z4', 'X4', 'Y4', 'Z5'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y5', equation_Y5, ['Z1', 'X1', 'Z2', 'X2', 'Y2', 'Z3', 'X3', 'Y3', 'Z4', 'X4', 'Y4', 'Z5','X5'], stats.norm(0, 0.1))

	X = ['X1', 'X2', 'X3', 'X4', 'X5']
	Y = ['Y3', 'Y4', 'Y5']

	return [scm, X, Y]

def Fulcher_FD(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_C1(noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		C1 = np.random.binomial(1, 0.6, num_samples)
		return C1 

	def equation_C2(C1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_C2 = inv_logit(1 + 0.5 * C1)
		return np.random.binomial(1, prob_C2) 

	def equation_C3(noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		C3 = np.random.binomial(1, 0.3, num_samples)
		return C3 

	def equation_X(C1, C2, C3, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( 0.5 + 0.2 * C1 + 0.4 * C2 + 0.5 * C1 * C2 + 0.2 * C3 )
		return np.random.binomial(1, prob_X)

	def equation_Z(C1, C2, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Z = inv_logit( 1 + X - 2*C1 + 2*C2 + 8*C1*C2 + noise )
		return np.random.binomial(1, prob_Z)

	def equation_Y(C1, C2, C3, Z, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( 1 + 2*X + 2*Z - 8*X*Z + 3*C1 + C2 + C1*C2 + C3 + noise )
		return np.random.binomial(1, prob_Y)


	scm = StructuralCausalModel()
	scm.add_observed_variable('C1', equation_C1, [], stats.norm(0, 0.1))
	scm.add_observed_variable('C2', equation_C2, ['C1'], stats.norm(0, 0.1))
	scm.add_observed_variable('C3', equation_C3, [], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['C1', 'C2', 'C3'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z', equation_Z, ['C1', 'C2', 'X'], stats.norm(0, 4))
	scm.add_observed_variable('Y', equation_Y, ['C1', 'C2', 'C3', 'Z', 'X'], stats.norm(0, 1))

	X = ['X']
	Y = ['Y']

	return [scm, X, Y]


def FD_SCM(seednum = None, dC = 4, dZ = 2):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_C(U_CX, U_CY, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		first_column = [2**(-abs(j - 0) - 1) for j in range(dC)]
		first_row = [2**(-abs(0 - k) - 1) for k in range(dC)]
		toeplitz_matrix = toeplitz(first_column, first_row)
		C = stats.multivariate_normal.rvs(mean = np.zeros(dC), cov = toeplitz_matrix, size=num_samples)
		for didx in range(dC):
			C[:,didx] = C[:,didx] + U_CX + U_CY
		return C

	def equation_X(U_XY, U_CX, C, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeff_X = [-(i) ** (-2) for i in range(1, dC + 1)]
		prob_X = inv_logit( np.dot(np.array(coeff_X), np.array(C).T) + U_XY - 0.5*U_CX + noise )
		return np.random.binomial(1, prob_X)

	def equation_Z(C, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		first_column = [2**(-abs(j - 0) - 1) for j in range(dZ)]
		first_row = [2**(-abs(0 - k) - 1) for k in range(dZ)]
		toeplitz_matrix = toeplitz(first_column, first_row)
		Z = stats.multivariate_normal.rvs(mean = np.zeros(dZ), cov = toeplitz_matrix, size=num_samples)
		
		coeff_Z = [-(i) ** (-2) for i in range(1, dC + 1)]
		for didx in range(dZ):
			prob_Z = inv_logit( np.dot(np.array(coeff_Z), np.array(C).T) + Z[:,didx] + (2*X-1) + noise )
			Z[:,didx] = np.random.binomial(1, prob_Z)
		return Z

	def equation_Y(U_XY, U_CY, C, Z, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeff_C = [-(i) ** (-2) for i in range(1, dC + 1)]
		coeff_Z = [-(i+1) ** (-2) for i in range(1, dZ + 1)]

		prob_Y = inv_logit( np.dot(np.array(coeff_Z), np.array(Z).T) + np.dot(np.array(coeff_C), np.array(C).T) + 1.5 * U_XY - 0.5*U_CY + noise )
		return np.random.binomial(1, prob_Y)


	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_CX', stats.norm(0, 1))
	scm.add_unobserved_variable('U_CY', stats.norm(0, 1))
	scm.add_unobserved_variable('U_XY', stats.norm(0, 1))
	scm.add_observed_variable('C', equation_C, ['U_CX', 'U_CY'], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['U_XY', 'U_CX', 'C'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z', equation_Z, ['C', 'X'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['U_XY', 'U_CY', 'C', 'Z'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']

	return [scm, X, Y]

def Napkin_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_W(U_WX, U_WY, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_W = inv_logit( U_WX + U_WY + noise )
		return prob_W
		# return np.random.binomial(1, prob_W)

	def equation_R(W, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		binary_W = np.round( inv_logit(W) )
		prob_R = inv_logit( binary_W*(2+noise) + (1-binary_W)*(-2-noise)  )
		return np.random.binomial(1, prob_R)

	def equation_X(R, U_WX, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( R*(2 + U_WX) + (1-R) * (-2 - U_WX))
		return np.random.binomial(1, prob_X)

	def equation_Y(X, U_WY, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( X*(2 + U_WY) + (1-X) * (-2 - U_WY))
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_WX', stats.norm(3, 1))
	scm.add_unobserved_variable('U_WY', stats.norm(-2, 1))
	scm.add_observed_variable('W', equation_W, ['U_WX', 'U_WY'], stats.norm(0, 0.1))
	scm.add_observed_variable('R', equation_R, ['W'], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['R', 'U_WX'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['X', 'U_WY'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']

	return [scm, X, Y]


def Napkin_SCM_dim(seednum = None, W_dim = 3):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	# Dynamically create a list of equation_Wi functions for each dimension of W
	def create_equation_Wi(i):
		def equation_Wi(U_WX, U_WY, noise, **kwargs):
			num_samples = kwargs.pop('num_sample')
			prob_Wi = inv_logit( U_WX + U_WY + noise )
			return prob_Wi
		return equation_Wi

	def equation_R(W_list, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')

		coeff = [1 if i % 2 == 0 else -1 for i in range(len(W_list))]  
		W_agg = np.dot(np.array(W_list).T, coeff)  # Compute dot product
		binary_W = np.round(inv_logit(W_agg))
		prob_R = inv_logit( binary_W*(2+noise) + (1-binary_W)*(-2-noise)  )
		return np.random.binomial(1, prob_R)

	def equation_X(R, U_WX, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( R*(2 + U_WX) + (1-R) * (-2 - U_WX))
		return np.random.binomial(1, prob_X)

	def equation_Y(X, U_WY, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( X*(2 + U_WY) + (1-X) * (-2 - U_WY))
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_WX', stats.norm(3, 1))
	scm.add_unobserved_variable('U_WY', stats.norm(-2, 1))

	# Add observed variables Wi using dynamically generated equations
	W_list = []
	for i in range(W_dim):
		equation_Wi = create_equation_Wi(i)  # Dynamically create equation_Wi
		W_name = f'W{i+1}'
		scm.add_observed_variable(W_name, equation_Wi, ['U_WX', 'U_WY'], stats.norm(0, 0.1))
		W_list.append(W_name)

	# Modify this line to correctly pass W_list during SCM computation
	def equation_R_wrapper(**kwargs):
		W_values = [kwargs[f'W{i+1}'] for i in range(W_dim)]  # Collect all Wi values as W_list
		return equation_R(W_values, **kwargs)


	# scm.add_observed_variable('W', equation_W, ['U_WX', 'U_WY'], stats.norm(0, 0.1))
	scm.add_observed_variable('R', equation_R_wrapper, W_list, stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['R', 'U_WX'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['X', 'U_WY'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']

	return [scm, X, Y]

def Napkin_FD_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_W(U_WX, U_WZ, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_W = inv_logit( U_WX + U_WZ + noise )
		return prob_W
		# return np.random.binomial(1, prob_W)

	def equation_R(W, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_R = inv_logit(W - 3 + noise) 
		return np.random.binomial(1, prob_R)

	def equation_X(U_WX, U_XY, R, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( 2 * R - 1 + 0.5 * (U_WX + U_XY) + noise)
		return np.random.binomial(1, prob_X)

	def equation_Z(U_WZ, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Z = inv_logit( 2*(2 * X - 1) + 0.5 * U_WZ + noise)
		return np.random.binomial(1, prob_Z)

	def equation_Y(U_XY, Z, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( 2*(2 *Z-1) + 0.5 * U_XY + noise )
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_WX', stats.norm(0, 1))
	scm.add_unobserved_variable('U_WZ', stats.norm(0, 1))
	scm.add_unobserved_variable('U_XY', stats.norm(0, 1))
	scm.add_observed_variable('W', equation_W, ['U_WX', 'U_WZ'], stats.norm(0, 0.1))
	scm.add_observed_variable('R', equation_R, ['W'], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['U_WX', 'U_XY', 'R'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z', equation_Z, ['U_WZ', 'X'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['U_XY', 'Z'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']

	return [scm, X, Y]

def Nested_Napkin_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_V1(U_V1X, U_V1V3, U_V1Y, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V1 = inv_logit( U_V1X + U_V1V3 + U_V1Y + noise )
		return prob_V1
		# return np.random.binomial(1, prob_V1)

	def equation_V2(U_V2V3, V1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V2 = inv_logit( U_V2V3 + V1 + noise )
		return np.random.binomial(1, prob_V2)

	def equation_V3(U_V1V3, U_V2V3, U_V3V5, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V3 = inv_logit(U_V1V3 + U_V2V3 + U_V3V5 + noise )
		return prob_V3
		# return np.random.binomial(1, prob_V3)

	def equation_V4(U_V4V5, V3, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V4 = inv_logit( U_V4V5 + V3 + noise)
		return np.random.binomial(1, prob_V4)

	def equation_V5(U_V3V5, U_V4V5, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V5 = inv_logit( U_V3V5 + U_V4V5 +  noise )
		return prob_V5
		# return np.random.binomial(1, prob_V5)

	def equation_X(U_V1X, V2, V4, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( 2 * (2*V2-1) + 0.3 * U_V1X + (2*V4-1) + noise )
		return np.random.binomial(1, prob_X)

	def equation_Y(U_V1Y, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( 2 * X -1  + 1.5 * U_V1Y + noise )
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_V1X', stats.norm(0, 1))
	scm.add_unobserved_variable('U_V1V3', stats.norm(0, 1))
	scm.add_unobserved_variable('U_V2V3', stats.norm(0, 1))
	scm.add_unobserved_variable('U_V3V5', stats.norm(0, 1))
	scm.add_unobserved_variable('U_V4V5', stats.norm(0, 1))
	scm.add_unobserved_variable('U_V1Y', stats.norm(0, 1))
	scm.add_observed_variable('V1', equation_V1, ['U_V1X', 'U_V1V3', 'U_V1Y'], stats.norm(0, 0.1))
	scm.add_observed_variable('V2', equation_V2, ['U_V2V3', 'V1'], stats.norm(0, 0.1))
	scm.add_observed_variable('V3', equation_V3, ['U_V1V3', 'U_V2V3', 'U_V3V5'], stats.norm(0, 0.1))
	scm.add_observed_variable('V4', equation_V4, ['U_V4V5', 'V3'], stats.norm(0, 0.1))
	scm.add_observed_variable('V5', equation_V5, ['U_V3V5', 'U_V4V5'], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['U_V1X', 'V2', 'V4'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['U_V1Y', 'X'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']

	return [scm, X, Y]

def Napkin_FD_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_W(U_WX, U_WZ, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_W = inv_logit( U_WX + U_WZ + noise )
		return prob_W
		# return np.random.binomial(1, prob_W)

	def equation_R(W, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_R = inv_logit(W - 3 + noise) 
		return np.random.binomial(1, prob_R)

	def equation_X(U_WX, U_XY, R, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( 2 * R - 1 + 0.5 * (U_WX + U_XY) + noise)
		return np.random.binomial(1, prob_X)

	def equation_Z(U_WZ, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Z = inv_logit( 2*(2 * X - 1) + 0.5 * U_WZ + noise)
		return np.random.binomial(1, prob_Z)

	def equation_Y(U_XY, Z, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( 2*(2 *Z-1) + 0.5 * U_XY + noise )
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_WX', stats.norm(0, 1))
	scm.add_unobserved_variable('U_WZ', stats.norm(0, 1))
	scm.add_unobserved_variable('U_XY', stats.norm(0, 1))
	scm.add_observed_variable('W', equation_W, ['U_WX', 'U_WZ'], stats.norm(0, 0.1))
	scm.add_observed_variable('R', equation_R, ['W'], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['U_WX', 'U_XY', 'R'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z', equation_Z, ['U_WZ', 'X'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['U_XY', 'Z'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']

	return [scm, X, Y]

def Napkin_FD_v2_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_X1(U_X1X2, U_X1V1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X1 = inv_logit( U_X1X2 + U_X1V1 + noise )
		return np.random.binomial(1, prob_X1)

	def equation_V2(X1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V2 = inv_logit( 2*X1 -1 + noise )
		return np.random.binomial(1, prob_V2)

	def equation_X2(U_X1X2, U_X2Y, V2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X2 = inv_logit( 2*V2 -1 + 0.5*(U_X1X2 + U_X2Y) + noise )
		return np.random.binomial(1, prob_X2)

	def equation_V1(U_X1V1, X2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V1 = inv_logit( 2*X2 -1 + 0.3 * U_X1V1 + noise )
		return np.random.binomial(1, prob_V1)

	def equation_V3(X2, V2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V3 = inv_logit( 2*X2 -1 + 0.5*(V2 -1) + noise )
		return np.random.binomial(1, prob_V3)

	def equation_Y(U_X2Y, V2, V3, V1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( 2*(2 *V2-1) + (2*V3*U_X2Y-1) + 0.5*(2*V1-1) + 0.5 * U_X2Y + noise )
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_X1X2', stats.norm(0, 1))
	scm.add_unobserved_variable('U_X1V1', stats.norm(0, 1))
	scm.add_unobserved_variable('U_X2Y', stats.norm(0, 1))
	scm.add_observed_variable('X1', equation_X1, ['U_X1X2', 'U_X1V1'], stats.norm(0, 0.1))
	scm.add_observed_variable('V2', equation_V2, ['X1'], stats.norm(0, 0.1))
	scm.add_observed_variable('X2', equation_X2, ['U_X1X2', 'U_X2Y', 'V2'], stats.norm(0, 0.1))
	scm.add_observed_variable('V1', equation_V1, ['U_X1V1', 'X2'], stats.norm(0, 0.1))
	scm.add_observed_variable('V3', equation_V3, ['X2', 'V2'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['U_X2Y', 'V2', 'V3', 'V1'], stats.norm(0, 0.1))

	X = ['X1', 'X2']
	Y = ['Y']

	return [scm, X, Y]

def Double_Napkin_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_V1(U_V1X, U_V1Y, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V1 = inv_logit( -2*U_V1X + U_V1Y + 1 + noise )
		return np.random.binomial(1, prob_V1)

	def equation_V2(U_V4V2, V3, V1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V2 = inv_logit( U_V4V2 + (2*V1-1) - 0.5*(2*V3-1) + noise )
		return np.random.binomial(1, prob_V2)

	def equation_V3(V4, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V3 = inv_logit( 2*V4 - 1 + noise )
		return np.random.binomial(1, prob_V3)

	def equation_V4(U_V4V2, U_V4X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_V4 = inv_logit( U_V4V2 + U_V4X + noise)
		return prob_V4
		# return np.random.binomial(1, prob_V4)

	def equation_X(U_V4X, U_V1X, V2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( 2 * (2*V2-1) + 0.3 * U_V4X + (2*U_V1X-1) + noise )
		return np.random.binomial(1, prob_X)

	def equation_Y(U_V1Y, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( 2 * X -1  + 1.5 * U_V1Y + noise )
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_V4V2', stats.norm(0, 1))
	scm.add_unobserved_variable('U_V4X', stats.norm(0, 1))
	scm.add_unobserved_variable('U_V1X', stats.norm(0, 1))
	scm.add_unobserved_variable('U_V1Y', stats.norm(0, 1))
	scm.add_observed_variable('V1', equation_V1, ['U_V1X', 'U_V1Y'], stats.norm(0, 0.1))
	scm.add_observed_variable('V2', equation_V2, ['U_V4V2', 'V3', 'V1'], stats.norm(0, 0.1))
	scm.add_observed_variable('V3', equation_V3, ['V4'], stats.norm(0, 0.1))
	scm.add_observed_variable('V4', equation_V4, ['U_V4V2', 'U_V4X'], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['U_V4X', 'U_V1X', 'V2'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['U_V1Y', 'X'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']

	return [scm, X, Y]

def Plan_ID_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_X1(U_X1Y, U_X1Z, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X1 = inv_logit( -2*U_X1Y + U_X1Z + 1 + noise )
		return np.random.binomial(1, prob_X1)

	def equation_R(X1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_R = inv_logit( 2*(2*X1-1) + noise )
		return np.random.binomial(1, prob_R)

	def equation_Z(U_X1Z, X1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Z = inv_logit( 2*X1 - 1 + 0.5* U_X1Z + noise )
		# return np.random.binomial(1, prob_Z)
		return prob_Z

	def equation_X2(Z, X1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X2 = inv_logit( 2*Z-1 + 2*(2*X1-1) + noise)
		return np.random.binomial(1, prob_X2)

	def equation_Y(U_X1Y, U_ZY, R, X2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( 2 * R -1  + 2*(2*X2-1) + 1.5 * U_X1Y + 0.5*U_ZY + noise )
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_X1Y', stats.norm(0, 1))
	scm.add_unobserved_variable('U_X1Z', stats.norm(0, 1))
	scm.add_unobserved_variable('U_ZY', stats.norm(0, 1))
	scm.add_observed_variable('X1', equation_X1, ['U_X1Y', 'U_X1Z'], stats.norm(0, 0.1))
	scm.add_observed_variable('R', equation_R, ['X1'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z', equation_Z, ['U_X1Z', 'X1'], stats.norm(0, 0.1))
	scm.add_observed_variable('X2', equation_X2, ['Z', 'X1'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['U_X1Y', 'U_ZY', 'R', 'X2'], stats.norm(0, 0.1))

	X = ['X1', 'X2']
	Y = ['Y']

	return [scm, X, Y]
	