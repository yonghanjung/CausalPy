import random 
import scipy.stats as stats
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.special import expit, softmax


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

def CCDDHNR2018_IRM(seednum=None, **kwargs):
    """
    Binary-treatment Interactive-Regression-Model (IRM) generator from
    Chernozhukov et al. (2018, App. P), refactored for the SCM framework.

    Optional keyword arguments (with defaults)
    ------------------------------------------
    d       : int, number of covariates          (default 20)
    theta   : float, causal effect               (default 0.5)
    R2_d    : float, targeted R² for treatment   (default 0.5)
    R2_y    : float, targeted R² for outcome     (default 0.5)
    corr    : float, base correlation ρ in Σ     (default 0.5)
    """

    # --- Model Parameters ---
    d = kwargs.get('d', 20)
    theta  = kwargs.get('theta', 0.5)
    R2_d   = kwargs.get('R2_d', 0.5)
    R2_y   = kwargs.get('R2_y', 0.5)
    corr   = kwargs.get('corr', 0.5)

    # --- Random-Seed Handling ---
    if seednum is not None:
        random.seed(int(seednum))
        np.random.seed(seednum)
    else:
        random.seed(123)
        np.random.seed(123)

    # --- Pre-calculate Constants and Vectors ---
    Sigma = toeplitz(corr ** np.arange(d))
    beta = 1.0 / (np.arange(1, d + 1) ** 2)
    beta_Sig_beta = beta @ Sigma @ beta
    c_d = np.sqrt((np.pi**2 / 3) * R2_d / ((1 - R2_d) * beta_Sig_beta))
    c_y = np.sqrt(R2_y / ((1 - R2_y) * beta_Sig_beta))

    # --- Structural Equations ---

    # Equation for covariates X (exogenous)
    def equation_X(noise, **eq_kwargs):
        num_samples = eq_kwargs.pop('num_sample')
        # Sigma is pre-calculated in the outer scope
        return stats.multivariate_normal.rvs(mean=np.zeros(d), cov=Sigma, size=num_samples)

    # Equation for treatment D (binary)
    def equation_D(X, noise, **eq_kwargs):
        # The noise argument is unused; randomness is from the binomial draw.
        # The random seed is set in the outer function.
        logits = c_d * (X @ beta)
        p = inv_logit(logits)
        return np.random.binomial(1, p)

    # Equation for outcome Y
    def equation_Y(D, X, noise, **eq_kwargs):
        # The noise argument corresponds to zeta ~ N(0,1) from the original paper.
        return theta * D + c_y * (X @ beta) + noise

    # --- SCM Construction ---
    scm = StructuralCausalModel()

    # Add variables to the SCM object
    scm.add_observed_variable('X', equation_X, [], stats.norm(0, 0.1))  # Placeholder noise
    scm.add_observed_variable('D', equation_D, ['X'], stats.norm(0, 0.1))  # Placeholder noise
    scm.add_observed_variable('Y', equation_Y, ['D', 'X'], stats.norm(0, 1))  # Noise zeta ~ N(0,1)

    # --- Define Treatment and Outcome variables ---
    D_vars = ['D']
    Y_vars = ['Y']

    return [scm, D_vars, Y_vars]

def CCDDHNR2018_PLR(seednum=None, d=20, **kwargs):
    """
    Partially-Linear-Regression (PLR) model from Chernozhukov et al. (2018),
    rewritten to conform to the SCM framework.
    """
    
    # Model parameters from kwargs
    theta = kwargs.get('theta', 0.5)
    a0 = kwargs.get('a0', 1.0)
    a1 = kwargs.get('a1', 0.25)
    b0 = kwargs.get('b0', 1.0)
    b1 = kwargs.get('b1', 0.25)
    s1 = kwargs.get('s1', 1.0)  # std dev for treatment noise
    s2 = kwargs.get('s2', 1.0)  # std dev for outcome noise

    # --- Random-seed handling ---
    if seednum is not None: 
        random.seed(int(seednum))
        np.random.seed(seednum)
    else:
        # Maintain original default seed behavior if no seed is provided
        random.seed(123)
        np.random.seed(123)

    # --- Structural Equations ---

    # Equation for covariates X (exogenous)
    def equation_X(noise, **eq_kwargs):
        num_samples = eq_kwargs.pop('num_sample')
        Sigma = toeplitz(0.7 ** np.arange(d))
        return stats.multivariate_normal.rvs(mean=np.zeros(d), cov=Sigma, size=num_samples)

    # Equation for treatment D
    def equation_D(X, noise, **eq_kwargs):
        # Nuisance function m0(x) for the treatment
        def m0(x):
            return a0 * x[..., 0] + a1 * np.exp(x[..., 2]) / (1.0 + np.exp(x[..., 2]))
        
        # D = m0(X) + noise (where noise ~ N(0, s1))
        return m0(X) + noise

    # Equation for outcome Y
    def equation_Y(D, X, noise, **eq_kwargs):
        # Nuisance function g0(x) for the outcome
        def g0(x):
            return b0 * np.exp(x[..., 0]) / (1.0 + np.exp(x[..., 0])) + b1 * x[..., 2]
        
        # Y = theta * D + g0(X) + noise (where noise ~ N(0, s2))
        return theta * D + g0(X) + noise

    # --- SCM Construction ---
    scm = StructuralCausalModel()
    
    # Add variables to the SCM
    # Placeholder noise for the exogenous variable X
    scm.add_observed_variable('X', equation_X, [], stats.norm(0, 0.1)) 
    # Treatment D depends on X, with noise Normal(0, s1)
    scm.add_observed_variable('D', equation_D, ['X'], stats.norm(0, s1))
    # Outcome Y depends on D and X, with noise Normal(0, s2)
    scm.add_observed_variable('Y', equation_Y, ['D', 'X'], stats.norm(0, s2))

    # --- Define Treatment and Outcome variables ---
    # Per the model's structure, D is the treatment and Y is the outcome.
    # Note: The covariate is 'X', while the treatment is 'D'.
    D_vars = ['D'] 
    Y_vars = ['Y']

    return [scm, D_vars, Y_vars]


def mSBD_SCM_JCI(seednum = None, d=4):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_C(noise, **kwargs):
		num_samples = kwargs.get('num_sample', None)
		return stats.multivariate_normal.rvs(mean = np.zeros(d), cov = np.eye(d), size=num_samples)

	def equation_X1(U_X1Z, C, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeffs = [1,-1,1] + [-(i - 2) ** (-2) for i in range(4,d+1)]
		prob = inv_logit( np.dot(np.array(coeffs), np.array(C).T ) + U_X1Z + noise )
		return np.random.binomial(1, prob)

	def equation_Z(U_X1Z, U_ZY, C, X1, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeffs = [-1,1,-1] + [(i - 2) ** (-2) for i in range(4,d+1)]
		prob = inv_logit( np.dot(np.array(coeffs), np.array(C).T) + (2*X1-1) * (U_X1Z + 2*U_ZY) + noise )
		return prob

	def equation_X2(X1, Z, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob = inv_logit( 2*X1-1 + Z + noise )
		return np.random.binomial(1, prob)

	def equation_Y(U_ZY, C, X1, X2, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeffs = [1,-1,1] + [-(i - 2) ** (-2) for i in range(4,d+1)]
		prob = inv_logit( np.dot(np.array(coeffs), np.array(C).T ) + (2*X1-1) * U_ZY + (2*X2-1) * U_ZY + noise )
		return np.random.binomial(1, prob)


	scm = StructuralCausalModel()
	scm.add_unobserved_variable('U_X1Z', stats.norm(0, 1))
	scm.add_unobserved_variable('U_ZY', stats.norm(0, 1))
	scm.add_observed_variable('C', equation_C, [], stats.norm(0, 0.1))
	scm.add_observed_variable('X1', equation_X1, ['U_X1Z', 'C'], stats.norm(0, 0.1))
	scm.add_observed_variable('Z', equation_Z, ['U_X1Z', 'U_ZY', 'C', 'X1'], stats.norm(0, 0.1))
	scm.add_observed_variable('X2', equation_X2, ['X1', 'Z'], stats.norm(0, 0.1))
	scm.add_observed_variable('Y', equation_Y, ['U_ZY', 'C', 'X1', 'X2'], stats.norm(0, 0.1))

	X = ['X1', 'X2']
	Y = ['Y']

	return [scm, X, Y]


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

def luedtke_2017_sim1_scm(seed=None, **kwargs):
    """
    Creates the Structural Causal Model for Simulation 1 from Luedtke et al. (2017).

    This function defines the structural equations for a 3-timepoint longitudinal
    study and returns the SCM object, treatment variables, and outcome variables.

    Args:
        seed (int, optional): A seed for the random number generator for
                              reproducibility. Defaults to None.

    Returns:
        list: A list containing [scm, treatment_vars, outcome_vars] where:
              - scm: The configured StructuralCausalModel object.
              - treatment_vars (list): A list of the treatment variable names.
              - outcome_vars (list): A list of the outcome variable names.
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Define Structural Equations ---
    num_samples = kwargs.get('num_sample', None)

    # Time-point t=1
    def equation_Z1(noise, **kwargs):
        # Z1 = stats.norm.rvs(loc=0, scale=1, size=num_samples)
        return noise

    def equation_X1(Z1, noise, **kwargs):
        prob = expit(Z1)
        return np.random.binomial(1, prob)

    # Time-point t=2
    def equation_Z2(noise, **kwargs):
        # Z2 = stats.norm.rvs(loc=0, scale=1, size=num_samples)
        return noise

    def equation_X2(Z2, X1, noise, **kwargs):
        prob = expit(Z2 + X1)
        return np.random.binomial(1, prob)

    # Time-point t=3
    def equation_Z3(Z1, X1, Z2, X2, noise, **kwargs):
        # L3 ~ Normal(L1*A2 + A1*L2 + L2*A2, 1).
        mean_Z3 = Z1 * X2 + X1 * Z2 + Z2 * X2
        return mean_Z3 + noise

    def equation_X3(Z3, X2, noise, **kwargs):
        # A3 ~ Bernoulli(expit(L3 + A2)).
        prob = expit(Z3 + X2)
        return np.random.binomial(1, prob)

    def equation_Y(Z2, X2, Z3, X3, noise, **kwargs):
        prob = expit(Z2 * X3 + X2 * Z3 + Z3 * X3)
        return np.random.binomial(1, prob)


    # --- SCM Construction ---
    scm = StructuralCausalModel()

    # Time-point 1
    scm.add_observed_variable('Z1', equation_Z1, [], stats.norm(0, 1))
    scm.add_observed_variable('X1', equation_X1, ['Z1'], stats.norm(0, 0.1)) # Placeholder noise

    # Time-point 2
    scm.add_observed_variable('Z2', equation_Z2, [], stats.norm(0, 1))
    scm.add_observed_variable('X2', equation_X2, ['Z2', 'X1'], stats.norm(0, 0.1))

    # Time-point 3
    scm.add_observed_variable('Z3', equation_Z3, ['Z1', 'X1', 'Z2', 'X2'], stats.norm(0, 1))
    scm.add_observed_variable('X3', equation_X3, ['Z3', 'X2'], stats.norm(0, 0.1))
    scm.add_observed_variable('Y', equation_Y, ['Z2', 'X2', 'Z3', 'X3'], stats.norm(0, 0.1))

    # --- Define Treatment and Outcome variables ---
    # The treatments are the time-varying actions A_t.
    # The final outcome of interest is Y3.
    treatments = ['X1', 'X2', 'X3']
    outcomes = ['Y']

    return [scm, treatments, outcomes]



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

def Canonical_FD_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_C(U_CX, U_CY, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		C = np.abs( np.random.normal(0,1,size=num_samples) )
		return C

	def equation_X(U_XY, U_CX, C, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( 0.5 * C +  U_XY + U_CX + np.abs( np.random.normal(0,1,size=num_samples) ) )
		return np.random.binomial(1, prob_X)

	def equation_Z(C, X, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Z = inv_logit( 0.5 * C +  2*X - 1 + np.abs( np.random.normal(0,1,size=num_samples) ) )
		return np.random.binomial(1, prob_Z)

	def equation_Y(U_XY, U_CY, C, Z, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( 0.5 * C +  2*Z - 1 + U_XY + U_CY + np.abs( np.random.normal(0,1,size=num_samples) ) )
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


def FD_SCM(seednum = None, dC = 3, dZ = 2):
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
			C_idx_val = inv_logit( C[:,didx] + U_CX + U_CY + 2 )
			C[:,didx] = np.random.binomial(1, C_idx_val)
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

def Bhattacharya2022_Fig2b_SCM(seednum=None, **kwargs):
    """
    Python implementation of the data generating process for the ADMG in Figure 2(b)
    [cite_start]from Bhattacharya, Nabi, and Shpitser (2022, Appendix G)[cite: 1128, 1129, 1130, 1131, 1132].
    """
    if seednum is not None:
        random.seed(int(seednum))
        np.random.seed(seednum)

    # --- Structural Equations for Hidden Variables (U) ---
    # These are defined by their distributions and added to the SCM.

    # --- Structural Equations for Observed Variables ---
    def equation_C1(noise, **kwargs): return np.random.binomial(1, 0.3, kwargs.get('num_sample'))
    def equation_C2(noise, **kwargs): return np.random.uniform(-1, 2, kwargs.get('num_sample'))
    def equation_C3(noise, **kwargs): return np.random.normal(1, 1, kwargs.get('num_sample'))

    def equation_X(C1, C2, C3, U1, U2, U3, noise, **kwargs):
        C4 = stats.norm.cdf(C3)
        C5 = C3**C1 + (1 - C1) * np.sin(np.abs(C3) * np.pi)
        C6 = C1 * C2 + np.abs(C3)
        logits = 0.5 + 0.9 * C4 - 0.5 * C5 + 0.2 * C6 + 0.3 * U1 - 0.8 * U2 + 0.8 * U3
        prob = expit(logits)
        return np.random.binomial(1, prob)

    def equation_M(C1, C2, C3, X, U4, U5, U6, noise, **kwargs):
        C4 = stats.norm.cdf(C3)
        C5 = C3**C1 + (1 - C1) * np.sin(np.abs(C3) * np.pi)
        C6 = C1 * C2 + np.abs(C3)
        logits = (0.5 - 0.7*C1 + 0.8*C2 - C3 - 1.2*X - 0.2*U4 + 0.5*U5 + 0.4*U6
                  + (1.5*C4 + 1.2*C5 + 0.6*C6) * X)
        prob = expit(logits)
        return np.random.binomial(1, prob)

    def equation_L(C1, C2, C3, M, X, U1, U2, U3, noise, **kwargs):
        # Note: L depends on T through M. Added T to parents for clarity.
        C4 = stats.norm.cdf(C3)
        C5 = C3**C1 + (1 - C1) * np.sin(np.abs(C3) * np.pi)
        C6 = C1 * C2 + np.abs(C3)
        logits = (-0.5*X + 0.8*C4 + 1.2*C5 - 0.6*C6 - 1.2*M + 0.3*U1 + 0.6*U2 - 0.4*U3
                  - (0.8*C4 + 1.5*C5 + 0.4*C6) * M)
        prob = expit(logits)
        return np.random.binomial(1, prob)

    def equation_Y(C1, C2, C3, X, L, U4, U5, U6, noise, **kwargs):
        C4 = stats.norm.cdf(C3)
        C5 = C3**C1 + (1 - C1) * np.sin(np.abs(C3) * np.pi)
        C6 = C1 * C2 + np.abs(C3)
        mean = (0.5 + 0.5*C4 - 2*C5 + 0.8*C6 + 0.5*X + 0.6*L - 0.6*U4 + 0.5*U5
                - 0.5*U6 + 1.3*C4*X + 2.3*C5*L + 2*C6*X*L + 1.2*X*L)
        return mean + noise

    # --- SCM Construction ---
    scm = StructuralCausalModel()
    # Add unobserved (hidden) variables
    scm.add_unobserved_variable('U1', stats.bernoulli(0.4))
    scm.add_unobserved_variable('U2', stats.uniform(0, 1.5))
    scm.add_unobserved_variable('U3', stats.norm(0, 1))
    scm.add_unobserved_variable('U4', stats.bernoulli(0.6))
    scm.add_unobserved_variable('U5', stats.uniform(-1, 2)) # scale = 1 - (-1) = 2
    scm.add_unobserved_variable('U6', stats.norm(0, 1.5))

    # Add observed variables
    scm.add_observed_variable('C1', equation_C1, [], stats.norm(0, 0.1)) # Placeholder noise
    scm.add_observed_variable('C2', equation_C2, [], stats.norm(0, 0.1))
    scm.add_observed_variable('C3', equation_C3, [], stats.norm(0, 0.1))
    scm.add_observed_variable('X', equation_X, ['C1', 'C2', 'C3', 'U1', 'U2', 'U3'], stats.norm(0, 0.1))
    scm.add_observed_variable('M', equation_M, ['C1', 'C2', 'C3', 'X', 'U4', 'U5', 'U6'], stats.norm(0, 0.1))
    scm.add_observed_variable('L', equation_L, ['C1', 'C2', 'C3', 'M', 'X', 'U1', 'U2', 'U3'], stats.norm(0, 0.1))
    scm.add_observed_variable('Y', equation_Y, ['C1', 'C2', 'C3', 'X', 'L', 'U4', 'U5', 'U6'], stats.norm(0, 1.5))
 
    X = ['X']
    Y = ['Y']
    return [scm, X, Y]


def Bhattacharya2022_Fig3_SCM(seednum=None, **kwargs):
    """
    Python implementation for the ADMG in Figure 3 from Bhattacharya et al. (2022).
    - Treatment is named 'X'.
    - Unmeasured confounders are named 'U_#'.
    """
    if seednum is not None:
        random.seed(int(seednum))
        np.random.seed(seednum)

    # --- Structural Equations for Observed Variables ---
    def equation_C11(noise, **kwargs): return np.random.normal(1, 1, kwargs.get('num_sample'))
    def equation_C12(noise, **kwargs): return np.random.uniform(-1, 1, kwargs.get('num_sample'))
    def equation_C21(noise, **kwargs): return np.random.normal(0, 1, kwargs.get('num_sample'))
    def equation_C22(noise, **kwargs): return np.random.binomial(1, 0.4, kwargs.get('num_sample'))

    def equation_X(C11, C12, C21, C22, U_1, U_2, U_3, noise, **kwargs):
        C3 = stats.norm.cdf(C11 * C12) + (1 - C12) * np.sin(np.abs(C11) * np.pi)
        C4 = (C21**C22) + (1 - C22) * np.sin(np.abs(C21) * np.pi)
        logits = (-0.5 + 0.9*C11 - 0.7*C12 + 0.6*C21 - 0.7*C22 + 0.3*U_1 - 0.5*U_2
                  + 0.4*U_3 + 1.6*C3 - 0.8*C4)
        return np.random.binomial(1, expit(logits))

    def equation_M(C21, C22, X, noise, **kwargs):
        C4 = (C21**C22) + (1 - C22) * np.sin(np.abs(C21) * np.pi)
        logits = -0.5 - 1.4*C21 + 1.3*C22 - 1.2*X + 2.2*C4*X - C4
        return np.random.binomial(1, expit(logits))

    def equation_L(C11, C12, C21, C22, M, U_1, U_2, U_3, noise, **kwargs):
        C3 = stats.norm.cdf(C11 * C12) + (1 - C12) * np.sin(np.abs(C11) * np.pi)
        C4 = (C21**C22) + (1 - C22) * np.sin(np.abs(C21) * np.pi)
        logits = (0.5 - 0.5*C11 - 0.4*C12 + 0.8*C21 + 0.9*C22 - 1.2*M + 0.3*U_1
                  + 0.6*U_2 - 0.4*U_3 - 1.8*C3*M - 1.5*C4*M + 1.2*C3 + 0.8*C4)
        return np.random.binomial(1, expit(logits))

    def equation_Y(C21, C22, L, noise, **kwargs):
        C4 = (C21**C22) + (1 - C22) * np.sin(np.abs(C21) * np.pi)
        mean = 0.5 + 0.7*C21 - 0.5*C22 + 1.6*L + 1.1*C4*L + 0.8*C4
        return mean + noise

    # --- SCM Construction ---
    scm = StructuralCausalModel()
    scm.add_unobserved_variable('U_1', stats.bernoulli(0.4))
    scm.add_unobserved_variable('U_2', stats.uniform(0, 1.5))
    scm.add_unobserved_variable('U_3', stats.norm(0, 1))

    scm.add_observed_variable('C11', equation_C11, [], stats.norm(0, 0.1))
    scm.add_observed_variable('C12', equation_C12, [], stats.norm(0, 0.1))
    scm.add_observed_variable('C21', equation_C21, [], stats.norm(0, 0.1))
    scm.add_observed_variable('C22', equation_C22, [], stats.norm(0, 0.1))
    scm.add_observed_variable('X', equation_X, ['C11', 'C12', 'C21', 'C22', 'U_1', 'U_2', 'U_3'], stats.norm(0, 0.1))
    scm.add_observed_variable('M', equation_M, ['C21', 'C22', 'X'], stats.norm(0, 0.1))
    scm.add_observed_variable('L', equation_L, ['C11', 'C12', 'C21', 'C22', 'M', 'U_1', 'U_2', 'U_3'], stats.norm(0, 0.1))
    scm.add_observed_variable('Y', equation_Y, ['C21', 'C22', 'L'], stats.norm(0, 1.5))

    treatment = ['X']
    outcome = ['Y']
    return [scm, treatment, outcome]


def Napkin_SCM(seednum = None):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_W(U_WX, U_WY, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_W = inv_logit( 3*U_WX * U_WY + noise + U_WX)
		return prob_W
		# return np.random.binomial(1, prob_W)

	def equation_R(W, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		binary_W = inv_logit(W) 
		prob_R = inv_logit( binary_W*(2+noise) + (1-binary_W)*(-2-noise) + 2*W  )
		return np.random.binomial(1, prob_R)

	def equation_X(R, U_WX, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_X = inv_logit( R*(2 + U_WX) + (1-R) * (-2 - U_WX) + 5*(2*R-1))
		return np.random.binomial(1, prob_X)

	def equation_Y(X, U_WY, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		prob_Y = inv_logit( X*(2 + U_WY) + (1-X) * (-2 - U_WY) + 2*U_WY*(2*X-1))
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

def Bhattacharya2022_Fig5_SCM(seednum=None, **kwargs):
    """
    Python implementation for the ADMG in Figure 5 from Bhattacharya et al. (2022).
    - Treatment is named 'X'.
    - Unmeasured confounders are named 'U_#'.
    """
    if seednum is not None:
        random.seed(int(seednum))
        np.random.seed(seednum)

    # --- Structural Equations ---
    def equation_C1(U_7, U_8, U_9, U_10, noise, **kwargs):
        return 0.4*U_7 - 0.1*U_8 + 0.6*U_9 + 0.8*U_10 + noise

    def equation_C2(U_7, U_8, U_9, U_10, noise, **kwargs):
        return -0.3*U_7 - 0.7*U_8 + 0.8*U_9 + 1.2*U_10 + noise

    def equation_R2(U_1, U_2, U_3, U_4, noise, **kwargs):
        return np.random.binomial(1, expit(-0.2 + 0.3*U_1 - 0.8*U_2 + 0.4*U_3 + 0.6*U_4))

    def equation_Z(U_1, U_2, U_5, U_6, noise, **kwargs):
        return np.random.binomial(1, expit(-0.5 + U_1 + 0.2*U_2 - 0.8*U_5 + 0.3*U_6))

    def equation_X(C1, C2, Z, U_3, U_4, noise, **kwargs):
        C3 = np.abs(C1 * C2)**0.5 + np.sin(np.abs(C1 + C2) * np.pi)
        C4 = stats.norm.cdf(C1)
        logits = 0.5 - 0.5*C1 + 0.5*C2 + 0.3*Z + 0.5*U_3 - 0.4*U_4 + 0.8*C3 - 1.3*C4
        return np.random.binomial(1, expit(logits))

    def equation_R1(X, U_5, U_6, noise, **kwargs):
        return np.random.binomial(1, expit(0.2 + 0.7*X - 0.6*U_5 - 0.6*U_6))

    def equation_M(R1, U_7, U_8, noise, **kwargs):
        return np.random.binomial(1, expit(0.5 - 0.8*R1 + 1.2*U_7 - 1.5*U_8))

    def equation_Y(C1, C2, X, M, R2, U_9, U_10, noise, **kwargs):
        C3 = np.abs(C1 * C2)**0.5 + np.sin(np.abs(C1 + C2) * np.pi)
        C4 = stats.norm.cdf(C1)
        mean = (-1 + 0.5*C1 + 0.2*C2 + 1.2*X + 0.8*R2 + 0.8*M + 0.2*U_9 - 0.4*U_10
                + 0.8*C3 - 1.2*C4 + M*X)
        return mean + noise

    # --- SCM Construction ---
    scm = StructuralCausalModel()
    ps = [0.4, 0.3, 0.4, 0.3, 0.3]
    for i, p in zip(range(1, 10, 2), ps): scm.add_unobserved_variable(f'U_{i}', stats.bernoulli(p))
    for i in range(2, 11, 2): scm.add_unobserved_variable(f'U_{i}', stats.norm(0, 1))

    scm.add_observed_variable('C1', equation_C1, ['U_7', 'U_8', 'U_9', 'U_10'], stats.norm(0, 1))
    scm.add_observed_variable('C2', equation_C2, ['U_7', 'U_8', 'U_9', 'U_10'], stats.norm(0, 1))
    scm.add_observed_variable('R2', equation_R2, ['U_1', 'U_2', 'U_3', 'U_4'], stats.norm(0, 0.1))
    scm.add_observed_variable('Z', equation_Z, ['U_1', 'U_2', 'U_5', 'U_6'], stats.norm(0, 0.1))
    scm.add_observed_variable('X', equation_X, ['C1', 'C2', 'Z', 'U_3', 'U_4'], stats.norm(0, 0.1))
    scm.add_observed_variable('R1', equation_R1, ['X', 'U_5', 'U_6'], stats.norm(0, 0.1))
    scm.add_observed_variable('M', equation_M, ['R1', 'U_7', 'U_8'], stats.norm(0, 0.1))
    scm.add_observed_variable('Y', equation_Y, ['C1', 'C2', 'X', 'M', 'R2', 'U_9', 'U_10'], stats.norm(0, 1))

    treatment = ['X']
    outcome = ['Y']
    return [scm, treatment, outcome]

def ConeCloud_15_SCM(seednum=None, **kwargs):
    """
    An example Structural Causal Model for the 15-node Cone Cloud graph
    from Figure 3b of Bhattacharya et al. (2022).

    - Treatment (Intervention): X = V10
    - Outcome: Y = V4
    - Observable variables are categorical with a domain size of 4.
    - Unmeasured confounders (U_*) represent the bidirected edges.
    """
    if seednum is not None:
        random.seed(int(seednum))
        np.random.seed(seednum)

    # Helper function to generate categorical variables
    def generate_categorical(logits, num_samples):
        if logits.ndim == 1:
            logits = np.tile(logits, (4, 1)).T
        probabilities = softmax(logits, axis=1)
        return np.array([np.random.choice(4, p=p_row) for p_row in probabilities])

    # --- Structural Equations ---
    # Central node
    def equation_V5(noise, **kwargs):
        num_samples = kwargs.get('num_sample')
        logits = np.tile(noise.reshape(-1, 1), (1, 4))
        return generate_categorical(logits, num_samples)

    # Nodes dependent on V5
    def equation_V3(V5, noise, **kwargs):
        logits = 0.8 * V5 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    def equation_V8(V5, noise, **kwargs):
        logits = 0.7 * V5 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    def equation_V6(V5, U_1_6, noise, **kwargs):
        logits = 0.5 * V5 + 1.2 * U_1_6 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    def equation_V7(V5, U_2_7, noise, **kwargs):
        logits = 0.6 * V5 + 1.1 * U_2_7 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    def equation_V9(V5, U_13_9, noise, **kwargs):
        logits = 0.4 * V5 + 1.3 * U_13_9 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    def equation_V11(V5, U_12_11, noise, **kwargs):
        logits = 0.55 * V5 + 1.4 * U_12_11 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    # Outer layer nodes
    def equation_V1(V3, U_1_6, noise, **kwargs):
        logits = 0.9 * V3 + 0.8 * U_1_6 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    def equation_V2(V3, U_2_7, noise, **kwargs):
        logits = 0.85 * V3 + 0.9 * U_2_7 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    def equation_V13(V8, U_13_9, noise, **kwargs):
        logits = 0.75 * V8 + 0.7 * U_13_9 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    def equation_V12(V8, U_12_11, noise, **kwargs):
        logits = 0.95 * V8 + 0.6 * U_12_11 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    # Treatment and Outcome variables
    def equation_X1(V9, V11, U_10_4, noise, **kwargs): # Previously V4
        logits = 0.7 * V9 + 0.6 * V11 + 2.0 * U_10_4 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    def equation_X2(V6, V7, U_10_4, noise, **kwargs): # Previously V10
        logits = 0.6 * V6 + 0.5 * V7 - 1.2 * U_10_4 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    def equation_X3(V12, V13, U_0_14, noise, **kwargs): # Previously V14
        logits = 0.5 * V12 + 0.3 * V13 + 1.5 * U_0_14 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    def equation_Y(V1, V2, U_0_14, noise, **kwargs): # Previously V0
        logits = 0.4 * V1 + 0.4 * V2 + 1.5 * U_0_14 + noise
        return generate_categorical(logits, kwargs.get('num_sample'))

    # --- SCM Construction ---
    scm = StructuralCausalModel()

    # Add unobserved variables for bidirected edges
    scm.add_unobserved_variable('U_0_14', stats.norm(0, 1))
    scm.add_unobserved_variable('U_10_4', stats.norm(0, 1))
    scm.add_unobserved_variable('U_1_6', stats.norm(0, 1))
    scm.add_unobserved_variable('U_2_7', stats.norm(0, 1))
    scm.add_unobserved_variable('U_13_9', stats.norm(0, 1))
    scm.add_unobserved_variable('U_12_11', stats.norm(0, 1))

    # Add observed variables in a plausible topological order
    scm.add_observed_variable('V5', equation_V5, [], stats.norm(0, 0.5))
    scm.add_observed_variable('V3', equation_V3, ['V5'], stats.norm(0, 0.5))
    scm.add_observed_variable('V8', equation_V8, ['V5'], stats.norm(0, 0.5))
    scm.add_observed_variable('V6', equation_V6, ['V5', 'U_1_6'], stats.norm(0, 0.5))
    scm.add_observed_variable('V7', equation_V7, ['V5', 'U_2_7'], stats.norm(0, 0.5))
    scm.add_observed_variable('V9', equation_V9, ['V5', 'U_13_9'], stats.norm(0, 0.5))
    scm.add_observed_variable('V11', equation_V11, ['V5', 'U_12_11'], stats.norm(0, 0.5))
    scm.add_observed_variable('V1', equation_V1, ['V3', 'U_1_6'], stats.norm(0, 0.5))
    scm.add_observed_variable('V2', equation_V2, ['V3', 'U_2_7'], stats.norm(0, 0.5))
    scm.add_observed_variable('V13', equation_V13, ['V8', 'U_13_9'], stats.norm(0, 0.5))
    scm.add_observed_variable('V12', equation_V12, ['V8', 'U_12_11'], stats.norm(0, 0.5))
    scm.add_observed_variable('X1', equation_X1, ['V9', 'V11', 'U_10_4'], stats.norm(0, 0.5))
    scm.add_observed_variable('X2', equation_X2, ['V6', 'V7', 'U_10_4'], stats.norm(0, 0.5))
    scm.add_observed_variable('X3', equation_X3, ['V12', 'V13', 'U_0_14'], stats.norm(0, 0.5))
    scm.add_observed_variable('Y', equation_Y, ['V1', 'V2', 'U_0_14'], stats.norm(0, 0.5))

    treatments = ['X1', 'X2', 'X3']
    outcomes = ['Y']
    return [scm, treatments, outcomes]


def Napkin_SCM_dim(seednum = None, d=5):
	if seednum is not None: 
		random.seed(int(seednum))
		np.random.seed(seednum)

	def equation_W(U_WX, U_WY, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		
		first_column = [2**(-abs(j - 0) - 1) for j in range(d)]
		first_row = [2**(-abs(0 - k) - 1) for k in range(d)]
		toeplitz_matrix = toeplitz(first_column, first_row)
		base_matrix = stats.multivariate_normal.rvs(mean = np.zeros(d), cov = toeplitz_matrix, size=num_samples)

		coef_U1 = np.random.randn(d)  # or use np.random.rand(d) for [0,1] uniform coefficients
		coef_U2 = np.random.randn(d)
		constants = np.random.randn(d)

		U_WX = np.asarray(U_WX).reshape(num_samples, 1)
		U_WY = np.asarray(U_WY).reshape(num_samples, 1)

		additional_matrix = (U_WX * coef_U1) + (U_WY * coef_U2) + constants

		W = base_matrix + additional_matrix

		return W

	def equation_R(W, noise, **kwargs):
		num_samples = kwargs.pop('num_sample')
		coeffs = [-(i) ** (-2) for i in range(1,d+1)]
		prob = inv_logit( np.dot(np.array(coeffs), np.array(W).T ) + noise )
		return np.random.binomial(1, prob)

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

if __name__ == "__main__":
    [scm, X, Y] = ConeCloud_15_SCM(num_sample = 10000, d= 100)
    sample_data = scm.generate_samples(100)