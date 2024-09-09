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

	def equation_C1(U_C, noise, **kwargs):
		return U_C + noise 

	def equation_C2(U_C, C1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		return stats.norm(0, 1).rvs() - 2*C1

	def equation_X(C1, C2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = 1 / (1 + np.exp(- (0.3 * C1) + 0.7*C2 + noise + 1))
		return np.random.binomial(1, prob_X)

	def equation_Y(C1, C2, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		linear_model = 2*(2 * X - 1) - 0.2* C1 + 0.5 * C2
		prob_Y = inv_logit( linear_model )
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_C', stats.norm(0, 1))
	scm.add_observed_variable('C1', equation_C1, ['U_C'], stats.norm(0, 0.1))
	scm.add_observed_variable('C2', equation_C2, ['U_C', 'C1'], stats.norm(0, 0.1))
	scm.add_observed_variable('X', equation_X, ['C1', 'C2'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['C1', 'C2', 'X'], stats.norm(0, 0.1))

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
		prob_R = inv_logit( 0.3 * W + noise)
		return np.random.binomial(1, prob_R)

	def equation_X(R, U_WX, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( 0.5 * R + 0.3 * U_WX + noise )
		return np.random.binomial(1, prob_X)

	def equation_Y(X, U_WY, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		linear_model = 3*(2 * X - 1) + 1.5 * U_WY + noise
		prob_Y = inv_logit( linear_model )
		return np.random.binomial(1, prob_Y)

	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_WX', stats.norm(0, 1))
	scm.add_unobserved_variable('U_WY', stats.norm(0, 1))
	scm.add_observed_variable('W', equation_W, ['U_WX', 'U_WY'], stats.norm(0, 0.1))
	scm.add_observed_variable('R', equation_R, ['W'], stats.norm(0, 0.1))
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
	