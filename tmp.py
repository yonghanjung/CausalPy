import pandas as pd
import numpy as np
from scipy.special import expit

def generate_kang_schafer_data(n_samples=1000, seed=None):
    """
    Generates a dataset based on the simulation design from Kang and Schafer (2007).

    This data generating process is a standard benchmark for evaluating causal
    inference estimators, particularly those dealing with propensity scores and
    double robustness. It features non-linear relationships and a known true
    average treatment effect of zero.

    Args:
        n_samples (int): The number of samples (observations) to generate.
        seed (int, optional): A random seed for reproducibility.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated data with the
                          following columns:
                          - 'z1', 'z2', 'z3', 'z4': The four covariates.
                          - 'treatment': The binary treatment assignment (0 or 1).
                          - 'outcome': The observed outcome variable.
                          - 'true_propensity': The true probability of treatment.
                          - 'Y0': The true potential outcome if not treated.
                          - 'Y1': The true potential outcome if treated.
                          The true Average Treatment Effect (ATE) is E[Y1 - Y0],
                          which is 0 in this simulation.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Generate four independent standard normal covariates
    z = np.random.normal(loc=0, scale=1, size=(n_samples, 4))
    
    # 2. Define the true propensity score model (probability of treatment)
    # This is a logistic function of the covariates.
    true_propensity = expit(-z[:, 0] + 0.5 * z[:, 1] - 0.25 * z[:, 2] - 0.1 * z[:, 3])

    # 3. Assign treatment based on the true propensity scores
    treatment = np.random.binomial(1, true_propensity)

    # 4. Define the true outcome model
    # The key feature is that the true outcome does NOT depend on the treatment.
    # Therefore, the true treatment effect is zero.
    error = np.random.normal(loc=0, scale=1, size=n_samples)
    y_true = 210 + 27.4 * z[:, 0] + 13.7 * z[:, 1] + 13.7 * z[:, 2] + 13.7 * z[:, 3] + error

    # Potential outcomes are the same because treatment has no effect on the outcome
    Y0 = y_true
    Y1 = y_true
    
    # Observed outcome is simply the true outcome
    outcome = y_true

    # 5. Assemble the dataset into a pandas DataFrame
    data = pd.DataFrame({
        'z1': z[:, 0],
        'z2': z[:, 1],
        'z3': z[:, 2],
        'z4': z[:, 3],
        'treatment': treatment,
        'outcome': outcome,
        'true_propensity': true_propensity,
        'Y0': Y0,
        'Y1': Y1
    })

    return data

# --- Example Usage ---
if __name__ == '__main__':
    # Generate a sample dataset
    sample_data = generate_kang_schafer_data(n_samples=5, seed=42)

    print("Generated Sample Data (Kang and Schafer, 2007):")
    print(sample_data)

    # You can use this generated data to test causal effect estimators.
    # For example, an unbiased estimator should, on average, produce a
    # treatment effect close to zero.
    large_dataset = generate_kang_schafer_data(n_samples=100000, seed=123)
    
    # A naive comparison of means will be biased due to confounding
    naive_effect = large_dataset[large_dataset['treatment'] == 1]['outcome'].mean() - \
                   large_dataset[large_dataset['treatment'] == 0]['outcome'].mean()
                   
    print(f"\nTrue Average Treatment Effect (ATE): 0.0")
    print(f"Naive (biased) estimate of ATE: {naive_effect:.4f}")

