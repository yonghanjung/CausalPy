import random 
import scipy.stats as stats
import numpy as np
import pandas as pd
from SCM import StructuralCausalModel  # Ensure generateSCM.py is in the same directory

def inv_logit(vec):
	return 1/(1+np.exp(-vec))

def BD_SCM(num_samples, num_truth = 500000, generate_submodel = False, seed = None):
	if seed is not None: 
		random.seed(int(seed))
		np.random.seed(seed)

	def equation_C1(U_C, noise, num_sample):
		return U_C + noise 

	def equation_C2(U_C, C1, noise, num_sample):
		return stats.norm(0, 1).rvs(size = num_sample) - 2*C1

	def equation_X(C1, C2, noise, num_sample):
		prob_X = 1 / (1 + np.exp(- (0.3 * C1) + 0.7*C2 + noise + 1))
		return np.random.binomial(1, prob_X, size = num_sample)

	def equation_Xcf(noise, num_sample):
		return np.random.binomial(1, 0.5, size = num_sample)

	def equation_Y(C1, C2, X, noise, num_sample):
		# linear_model = (2 * X - 1) + (2 * X - 1) * C1 + 0.5 * C2 * (2 * X - 1) + noise 
		linear_model = 2*(2 * X - 1) - 0.2* C1 + 0.5 * C2
		# return linear_model
		prob_Y = inv_logit( linear_model )
		return prob_Y
		# return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_C', stats.norm(0, 1))
	scm.add_observed_variable('C1', equation_C1, ['U_C'], stats.norm(0, 0.1), num_samples)
	scm.add_observed_variable('C2', equation_C2, ['U_C', 'C1'], stats.norm(0, 0.1), num_samples)
	scm.add_observed_variable('X', equation_X, ['C1', 'C2'], stats.norm(0, 0.1), num_samples)
	scm.add_observed_variable('Y', equation_Y, ['C1', 'C2', 'X'], stats.norm(0, 0.1), num_samples)

	G = scm.graph 
	if seed is not None: 
		Data = scm.generate_samples(num_samples, seed)
	else:
		Data = scm.generate_samples(num_samples)

	if not generate_submodel:
		Data_cf = Data.copy()

	else:
		scm_sub = StructuralCausalModel()
		scm_sub.add_unobserved_variable('U_C', stats.norm(0, 1))
		scm_sub.add_observed_variable('C1', equation_C1, ['U_C'], stats.norm(0, 0.1), num_truth)
		scm_sub.add_observed_variable('C2', equation_C2, ['U_C', 'C1'], stats.norm(0, 0.1), num_truth)
		scm_sub.add_observed_variable('X', equation_Xcf, [], stats.norm(0, 0.1), num_truth)
		scm_sub.add_observed_variable('Y', equation_Y, ['C1', 'C2', 'X'], stats.norm(0, 0.1), num_truth)
		
		if seed is not None: 
			Data_cf = scm_sub.generate_samples(num_truth, seed)
		else:
			Data_cf = scm_sub.generate_samples(num_truth)	
	
	columns_to_drop = [col for col in Data.columns if col.startswith('U')]
	Data = Data.drop(columns=columns_to_drop)
	Data_cf = Data_cf.drop(columns=columns_to_drop)
	return [scm, G, Data, Data_cf]


def mSBD_SCM(num_samples, num_truth = 500000, generate_submodel = False, seed = None):
	if seed is not None: 
		random.seed(int(seed))
		np.random.seed(seed)

	def equation_Z1(U_Z1Y2, noise, num_sample):
		return U_Z1Y2 + noise 

	def equation_X1(U_X1X2, Z1, noise, num_sample):
		linear_model = (0.3 * Z1) + 0.7*U_X1X2 + noise + 1
		prob = inv_logit(linear_model)
		return np.random.binomial(1, prob, size = num_sample)

	def equation_Y1(U_Y1X2, Z1, X1, noise, num_sample):
		linear_model = (0.3 * Z1) + 0.7*U_Y1X2 + noise + 3*(2*X1-1)
		prob = inv_logit(linear_model)
		return np.random.binomial(1, prob, size = num_sample)

	def equation_Z2(Z1, Y1, noise, num_sample):
		linear_model = (0.3 * Z1) + 3*(2*Y1-1) + noise 
		return linear_model

	def equation_X2(U_Y1X2, U_X1X2, Z2, Y1, noise, num_sample):
		linear_model = (0.3 * Z2) + 0.7*U_Y1X2 - U_X1X2 + 3*(2*Y1-1)
		prob = inv_logit(linear_model)
		return np.random.binomial(1, prob, size = num_sample)

	def equation_Y2(U_Z1Y2, Z2, X2, noise, num_sample):
		linear_model = (0.3 * Z2) + 0.7*U_Z1Y2 + noise + 3*(2*X2-1)
		prob = inv_logit(linear_model)
		return np.random.binomial(1, prob, size = num_sample)

	def equation_X_cf(noise, num_sample):
		return np.random.binomial(1, 0.5, size = num_sample)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_Z1Y2', stats.norm(0, 1))
	scm.add_unobserved_variable('U_X1X2', stats.norm(0, 1))
	scm.add_unobserved_variable('U_Y1X2', stats.norm(0, 1))
	scm.add_observed_variable('Z1', equation_Z1, ['U_Z1Y2'], stats.norm(0, 0.1), num_samples)
	scm.add_observed_variable('X1', equation_X1, ['U_X1X2', 'Z1'], stats.norm(0, 0.1), num_samples)
	scm.add_observed_variable('Y1', equation_Y1, ['U_Y1X2', 'Z1', 'X1'], stats.norm(0, 0.1), num_samples)
	scm.add_observed_variable('Z2', equation_Z2, ['Z1', 'Y1'], stats.norm(0, 0.1), num_samples)
	scm.add_observed_variable('X2', equation_X2, ['U_Y1X2', 'U_X1X2', 'Z2', 'Y1'], stats.norm(0, 0.1), num_samples)
	scm.add_observed_variable('Y2', equation_Y2, ['U_Z1Y2', 'Z2', 'X2'], stats.norm(0, 0.1), num_samples)

	G = scm.graph 
	if seed is not None: 
		Data = scm.generate_samples(num_samples, seed)
	else:
		Data = scm.generate_samples(num_samples)

	if not generate_submodel:
		Data_cf = Data.copy()

	else:
		sub_scm = StructuralCausalModel()
		sub_scm.add_unobserved_variable('U_Z1Y2', stats.norm(0, 1))
		sub_scm.add_unobserved_variable('U_X1X2', stats.norm(0, 1))
		sub_scm.add_unobserved_variable('U_Y1X2', stats.norm(0, 1))
		sub_scm.add_observed_variable('Z1', equation_Z1, ['U_Z1Y2'], stats.norm(0, 0.1), num_truth)
		sub_scm.add_observed_variable('X1', equation_X_cf, [], stats.norm(0, 0.1), num_truth)
		sub_scm.add_observed_variable('Y1', equation_Y1, ['U_Y1X2', 'Z1', 'X1'], stats.norm(0, 0.1), num_truth)
		sub_scm.add_observed_variable('Z2', equation_Z2, ['Z1', 'Y1'], stats.norm(0, 0.1), num_truth)
		sub_scm.add_observed_variable('X2', equation_X_cf, [], stats.norm(0, 0.1), num_truth)
		sub_scm.add_observed_variable('Y2', equation_Y2, ['U_Z1Y2', 'Z2', 'X2'], stats.norm(0, 0.1), num_truth)
		
		if seed is not None: 
			Data_cf = sub_scm.generate_samples(num_truth, seed)
		else:
			Data_cf = sub_scm.generate_samples(num_truth)	
	
	columns_to_drop_Data = [col for col in Data.columns if col.startswith('U')]
	Data = Data.drop(columns=columns_to_drop_Data)
	
	columns_to_drop_Data_Cf = [col for col in Data_cf.columns if col.startswith('U')]
	Data_cf = Data_cf.drop(columns=columns_to_drop_Data_Cf)
	return [scm, G, Data, Data_cf]

def random_SCM(num_unobserved, num_observed, num_treatments, num_outcomes, num_samples):
	scm = StructuralCausalModel()
	scm.generate_random_scm(num_unobserved=num_unobserved, num_observed=num_observed, num_treatments=num_treatments, num_outcomes=num_outcomes)
	G = scm.graph 
	D = scm.generate_samples(num_samples)
	return [scm, G, D]


def FD_SCM(num_samples):
	def equation_C(U_CX, U_CY, noise=0):
		return stats.norm(0, 1).rvs() + U_CX + U_CY

	def equation_X(U_XY, U_CX, C, noise=0):
		prob_X = 1 / (1 + np.exp(- (0.3 * C) + U_XY - 0.5*U_CX))
		return np.random.binomial(1, prob_X)

	def equation_Z(C,X, noise=0):
		prob_Z = 1 / (1 + np.exp(- (0.3 * X) + C))
		return np.random.binomial(1, prob_Z)

	def equation_Y(U_XY, U_CY, C, Z, noise):
		return 2 * X + C + 1.5 * U_XY - 0.5*U_CY + noise

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_CX', stats.norm(0, 1))
	scm.add_unobserved_variable('U_CY', stats.norm(0, 1))
	scm.add_unobserved_variable('U_XY', stats.norm(0, 1))
	scm.add_observed_variable('C', equation_X, ['U_CX', 'U_CY'], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['U_XY', 'U_CX', 'C'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z', equation_Z, ['C', 'X'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['U_XY', 'U_CY', 'C', 'Z'], stats.norm(0, 0.1))

	G = scm.graph 
	D = scm.generate_samples(num_samples)
	return [scm, G, D]


def Napkin_SCM(num_samples, num_truth = 500000, generate_submodel = False, seed = None):
	def equation_W(U_WX, U_WY, noise, num_sample):
		return stats.norm(0, 1).rvs() + U_WX + U_WY

	def equation_R(W, noise, num_sample):
		prob_R = 1 / (1 + np.exp(- (0.3 * W)))
		return np.random.binomial(1, prob_R)

	def equation_Xcf(noise, num_sample):
		return np.random.binomial(1, 0.5, size = num_sample)

	def equation_X(R, U_WX, noise, num_sample):
		treatment_prob = 1 / (1 + np.exp(- (0.5 * R + 0.3 * U_WX)))
		return np.random.binomial(1, treatment_prob, size = num_sample)

	def equation_Y(X, U_WY, noise, num_sample):
		linear_model = 3*(2 * X - 1) + 1.5 * U_WY + noise
		prob_Y = inv_logit( linear_model )
		return np.random.binomial(1, prob_Y, size = num_sample)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_WX', stats.norm(0, 1))
	scm.add_unobserved_variable('U_WY', stats.norm(0, 1))
	scm.add_observed_variable('W', equation_W, ['U_WX', 'U_WY'], stats.norm(0, 0.1), num_samples)
	scm.add_observed_variable('R', equation_R, ['W'], stats.norm(0, 0.1), num_samples)
	scm.add_observed_variable('X', equation_X, ['R', 'U_WX'], stats.norm(0, 0.1), num_samples)
	scm.add_observed_variable('Y', equation_Y, ['X', 'U_WY'], stats.norm(0, 0.1), num_samples)

	G = scm.graph 
	if seed is not None: 
		Data = scm.generate_samples(num_samples, seed)
	else:
		Data = scm.generate_samples(num_samples)

	if not generate_submodel:
		Data_cf = Data.copy()

	else:
		scm_sub = StructuralCausalModel()
		scm_sub.add_unobserved_variable('U_WX', stats.norm(0, 1))
		scm_sub.add_unobserved_variable('U_WY', stats.norm(0, 1))
		scm_sub.add_observed_variable('W', equation_W, ['U_WX', 'U_WY'], stats.norm(0, 0.1), num_truth)
		scm_sub.add_observed_variable('R', equation_R, ['W'], stats.norm(0, 0.1), num_truth)
		scm_sub.add_observed_variable('X', equation_Xcf, [], stats.norm(0, 0.1), num_truth)
		scm_sub.add_observed_variable('Y', equation_Y, ['X', 'U_WY'], stats.norm(0, 0.1), num_truth)
		
		if seed is not None: 
			Data_cf = scm_sub.generate_samples(num_truth, seed)
		else:
			Data_cf = scm_sub.generate_samples(num_truth)	
	
	columns_to_drop = [col for col in Data.columns if col.startswith('U')]
	Data = Data.drop(columns=columns_to_drop)
	Data_cf = Data_cf.drop(columns=columns_to_drop)
	return [scm, G, Data, Data_cf]


def Napkin_FD_SCM(num_samples):
	if seed is not None: 
		random.seed(int(seed))
		np.random.seed(seed)

	def equation_W(U_WX, U_WZ, noise=0):
		return stats.norm(0, 1).rvs() + U_WX + U_WY

	def equation_R(W, noise=0):
		prob_R = 1 / (1 + np.exp(- (0.3 * W)))
		return np.random.binomial(1, prob_R)

	def equation_X(U_WX, U_XY, R, noise):
		treatment_prob = 1 / (1 + np.exp(- (0.5 * R + 0.3 * U_WX + U_XY)))
		return np.random.binomial(1, treatment_prob)

	def equation_Z(U_WZ, X, noise=0):
		prob_Z = 1 / (1 + np.exp(- (0.3 * X) + U_WZ))
		return np.random.binomial(1, prob_Z)

	def equation_Y(U_XY, Z, noise):
		return 2 * Z + 1.5 * U_XY + noise

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_WX', stats.norm(0, 1))
	scm.add_unobserved_variable('U_WZ', stats.norm(0, 1))
	scm.add_unobserved_variable('U_XY', stats.norm(0, 1))
	scm.add_observed_variable('W', equation_W, ['U_WX', 'U_WZ'], stats.norm(0, 0.1))
	scm.add_observed_variable('R', equation_R, ['W'], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['U_WX', 'U_XY', 'R'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z', equation_X, ['U_WZ', 'X'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['U_XY', 'Z'], stats.norm(0, 0.1))

	G = scm.graph 
	D = scm.generate_samples(num_samples)
	return [scm, G, D]


def Tian_SCM(num_samples):
	def equation_V1(U_V1X, U_V1V3, U_V1Y, noise=0):
		return stats.norm(0, 1).rvs() + U_V1X + U_V1V3 + U_V1Y

	def equation_V2(U_V2V3, V1, noise=0):
		return stats.norm(0, 1).rvs() + U_V2V3 + V1

	def equation_V3(U_V1V3, U_V2V3, U_V3V5, noise=0):
		return stats.norm(0, 1).rvs() + U_V1V3 + U_V2V3 + U_V3V5

	def equation_V4(U_V4V5, V3, noise=0):
		return stats.norm(0, 1).rvs() + U_V4V5 + V3

	def equation_V5(U_V3V5, U_V4V5, noise=0):
		return stats.norm(0, 1).rvs() + U_V1X + U_V1V3 + U_V1Y

	def equation_X(U_V1X, V2, V4, noise):
		treatment_prob = 1 / (1 + np.exp(- (0.5 * V2 + 0.3 * U_V1X + V4)))
		return np.random.binomial(1, treatment_prob)

	def equation_Y(U_V1Y, X, noise):
		return 2 * X -1  + 1.5 * U_V1Y + noise

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

	G = scm.graph 
	X = ['X']  # Replace with actual treatment variables
	Y = ['Y']  # Replace with actual outcome variables
	D = scm.generate_samples(num_samples)
	return [scm, G, D]
	