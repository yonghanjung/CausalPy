import random 
import scipy.stats as stats
import numpy as np
import pandas as pd

from SCM import StructuralCausalModel  # Ensure generateSCM.py is in the same directory

def inv_logit(vec):
	return 1/(1+np.exp(-vec))

def BD_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_C(U_C, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		return U_C + noise 

	def equation_X(C, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( 0.3 * C + 1 + noise)
		return np.random.binomial(1, prob_X)

	def equation_Y(C, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( 2*(2 * X - 1)*C + 0.5 * C + (2*X - 1) + noise )
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_C', stats.norm(0, 1))
	scm.add_observed_variable('C', equation_C, ['U_C'], stats.norm(0, 0.1))
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

def Dukes_Vansteelandt_Farrel(seednum = None, d=200):
	# Inference for treatment effect parameters in potentially misspecified high-dimensional models
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_Z(**kwargs):
		num_samples = kwargs.pop('num_sample')
		return stats.multivariate_normal.rvs(mean=[0,0], cov=[[1, 0.8], [0.8, 1]], size=num_samples)
	
	scm = StructuralCausalModel()
	scm.add_observed_variable('Z', equation_Z, [], stats.norm(0, 0.1))
	

def Kang_Schafer_dim(seednum = None, d=4):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def create_equation_Zi(i):
		def equation_Zi(**kwargs):
			num_samples = kwargs.get('num_sample', None)
			Zi = np.random.normal(0,1,num_samples)
			return Zi
		return equation_Zi

	def equation_X(Z_list, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')

		# coeff = [27.4] + [13.7] * (len(Z_list)-1) 
		coeff = [(-1)**n * 2**(-n) for n in range(1, d + 1)]
		X_agg = np.dot(np.array(Z_list).T, coeff)  # Compute dot product
		prob_X = inv_logit(X_agg)
		return np.random.binomial(1, prob_X)

	def equation_Y(Z_list, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')

		coeff = [27.4] + [13.7] * (len(Z_list) - 1)
		Y_agg = np.dot(np.array(Z_list).T, coeff)  # Compute dot product
		Y = 210 + X*Y_agg + noise
		return Y

	scm = StructuralCausalModel()
	
	# Add observed variables Zi using dynamically generated equations
	Z_list = []
	for i in range(d):
		equation_Zi = create_equation_Zi(i)  # Dynamically create equation_Wi
		Z_name = f'Z{i+1}'
		scm.add_observed_variable(Z_name, equation_Zi, [], stats.norm(0, 0.1))
		Z_list.append(Z_name)

		
	# Modify this line to correctly pass Z_list during SCM computation
	def equation_X_wrapper(**kwargs):
		Z_values = [kwargs[f'Z{i+1}'] for i in range(d)]  # Collect all Wi values as W_list
		return equation_X(Z_values, **kwargs)

	# Modify this line to correctly pass Z_list during SCM computation
	def equation_Y_wrapper(**kwargs):
		Z_values = [kwargs[f'Z{i+1}'] for i in range(d)]  # Collect all Wi values as W_list
		return equation_Y(Z_values, **kwargs)


	# scm.add_observed_variable('W', equation_W, ['U_WX', 'U_WY'], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X_wrapper, Z_list, stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y_wrapper, Z_list + ['X'], stats.norm(0, 0.1))

	X = ['X']
	Y = ['Y']
	return [scm, X, Y]


def mSBD_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_Z1(U_Z1Y2, noise, **kwargs):
		return U_Z1Y2 + noise 

	def equation_X1(U_X1X2, Z1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		linear_model = (0.3 * Z1) + 0.7*U_X1X2 + noise + 1
		prob = inv_logit(linear_model)
		return np.random.binomial(1, prob)

	def equation_Y1(U_Y1X2, Z1, X1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		linear_model = (0.3 * Z1) + 0.7*U_Y1X2 + noise + 3*(2*X1-1)
		prob = inv_logit(linear_model)
		return np.random.binomial(1, prob)

	def equation_Z2(Z1, X1, Y1, noise, **kwargs):
		linear_model = (0.3 * Z1) -2*(2*X1 -1) + 3*(2*Y1-1) + noise 
		return linear_model

	def equation_X2(U_Y1X2, U_X1X2, Z1, X1, Y1, Z2,  noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		linear_model = (0.3 * Z2) + 0.7*U_Y1X2 - U_X1X2 + 2*(2*(Z1 + X1 + Y1)-1)
		prob = inv_logit(linear_model)
		return np.random.binomial(1, prob)

	def equation_Y2(U_Z1Y2, Z1, X1, Y1, Z2, X2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		linear_model = (0.3 * Z2) + 0.7*U_Z1Y2 - 2*(2*X2-1) + 1*(Z1 + X1 + Y1 + Z2) + noise 
		prob = inv_logit(linear_model)
		return np.random.binomial(1, prob)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_Z1Y2', stats.norm(0, 1))
	scm.add_unobserved_variable('U_X1X2', stats.norm(0, 1))
	scm.add_unobserved_variable('U_Y1X2', stats.norm(0, 1))
	scm.add_observed_variable('Z1', equation_Z1, ['U_Z1Y2'], stats.norm(0, 0.1))
	scm.add_observed_variable('X1', equation_X1, ['U_X1X2', 'Z1'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y1', equation_Y1, ['U_Y1X2', 'Z1', 'X1'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z2', equation_Z2, ['Z1', 'X1', 'Y1'], stats.norm(0, 0.1))
	scm.add_observed_variable('X2', equation_X2, ['U_Y1X2', 'U_X1X2', 'Z1', 'X1', 'Y1', 'Z2'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y2', equation_Y2, ['U_Z1Y2', 'Z1', 'X1', 'Y1', 'Z2', 'X2'], stats.norm(0, 0.1))	

	X = ['X1', 'X2']
	Y = ['Y1', 'Y2']

	return [scm, X, Y]

def FD_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_C(U_CX, U_CY, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_C = inv_logit( U_CX + U_CY + noise )
		return np.random.binomial(1, prob_C)
		# return U_CX + U_CY + noise 

	def equation_X(U_XY, U_CX, C, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit(- (0.3 * C) + U_XY - 0.5*U_CX + noise)
		return np.random.binomial(1, prob_X)

	def equation_Z(C,X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Z = inv_logit(0.3 * (2*X-1) + C + noise)
		return np.random.binomial(1, prob_Z)

	def equation_Y(U_XY, U_CY, C, Z, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( 2*Z - 1 + C + 1.5 * U_XY - 0.5*U_CY + noise)
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
	