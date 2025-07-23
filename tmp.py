import pandas as pd
import numpy as np
import time
import signal
import sys
from collections import defaultdict
from itertools import product
from sklearn.linear_model import LogisticRegression

# --- Timeout Handling ---
class TimeoutException(Exception): pass
def timeout_handler(signum, frame): raise TimeoutException

# --- Data Generation and Method Implementations (Unchanged) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_data_with_known_truth(n, d, betas, gammas):
    z_cols = [f'Z{i+1}' for i in range(d)]
    df = pd.DataFrame(np.random.randint(0, 2, size=(n, d)), columns=z_cols)
    z_matrix = df[z_cols].values
    linear_term_x = gammas['intercept'] + z_matrix @ gammas['z']
    df['X'] = np.random.binomial(1, sigmoid(linear_term_x))
    linear_term_y = betas['intercept'] + betas['x'] * df['X'] + z_matrix @ betas['z']
    df['Y'] = np.random.binomial(1, sigmoid(linear_term_y))
    return df

# Note: calculate_true_value is no longer used but kept for reference
def calculate_true_value(d, x_val, betas):
    true_value = 0.0
    p_z = 0.5 ** d
    for z in product([0, 1], repeat=d):
        e_y_xz = sigmoid(betas['intercept'] + betas['x'] * x_val + np.array(z) @ betas['z'])
        true_value += e_y_xz * p_z
    return true_value

def naive_summation(df, x_val, z_cols):
    n, d = len(df), len(z_cols)
    if n == 0: return 0.0
    p_z_counts, e_yxz_sums, e_yxz_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in df.itertuples(index=False):
        z = tuple(getattr(r, col) for col in z_cols)
        p_z_counts[z] += 1
        key = (r.X, z); e_yxz_sums[key] += r.Y; e_yxz_counts[key] += 1
    total = 0.0
    for z in product([0, 1], repeat=d):
        p_z = p_z_counts.get(z, 0) / n
        if p_z == 0: continue
        key = (x_val, z); e_y_xz = e_yxz_sums.get(key, 0) / e_yxz_counts.get(key, 1)
        total += e_y_xz * p_z
    return total

def empirical_summation(df, x_val, z_cols):
    n = len(df)
    if n == 0: return 0.0
    p_z_counts, e_yxz_sums, e_yxz_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in df.itertuples(index=False):
        z = tuple(getattr(r, col) for col in z_cols)
        p_z_counts[z] += 1
        key = (r.X, z); e_yxz_sums[key] += r.Y; e_yxz_counts[key] += 1
    total = 0.0
    for z, count in p_z_counts.items():
        p_z = count / n
        key = (x_val, z); e_y_xz = e_yxz_sums.get(key, 0) / e_yxz_counts.get(key, 1)
        total += e_y_xz * p_z
    return total

def modeling_summation(df, x_val, z_cols):
    if len(df) == 0: return 0.0
    features = ['X'] + z_cols
    model = LogisticRegression(solver='liblinear', C=10)
    model.fit(df[features], df['Y'])
    df_pred = df.copy(); df_pred['X'] = x_val
    return np.mean(model.predict_proba(df_pred[features])[:, 1])

def time_method_with_timeout(func, timeout_seconds, *args, **kwargs):
    if sys.platform == "win32":
        start = time.perf_counter(); val = func(*args, **kwargs); end = time.perf_counter()
        return val, end - start
    signal.signal(signal.SIGALRM, timeout_handler); signal.alarm(timeout_seconds)
    start = time.perf_counter()
    try:
        val = func(*args, **kwargs); elapsed = time.perf_counter() - start
        return val, elapsed
    except TimeoutException: return "Timed Out", f"> {timeout_seconds}"
    finally: signal.alarm(0)

def run_comparison(d, n_samples, timeout):
    print(f"\n--- Running Comparison for d={d}, n={n_samples} (Timeout={timeout}s) ---")
    
    np.random.seed(42)
    betas = {'intercept': -0.5, 'x': 0.8, 'z': np.random.uniform(-0.5, 0.5, d)}
    gammas = {'intercept': 0.2, 'z': np.random.uniform(-0.3, 0.3, d)}

    df_sample = generate_data_with_known_truth(n_samples, d, betas, gammas)
    z_cols = [f'Z{i+1}' for i in range(d)]

    # 1. Establish ground truth using the modeling method
    # Run it without a timeout as it's our baseline.
    ground_truth = modeling_summation(df_sample, 1, z_cols)
    print(f"Ground Truth (from Modeling Method): {ground_truth:.4f}")

    # 2. Test each method and compare to the new ground truth
    methods = {"Naive Summation": naive_summation, "Empirical Summation": empirical_summation}
    results = {}
    
    # Store the modeling result first
    results['Modeling (g-comp)'] = {'Value': ground_truth, 'Time (s)': "N/A (Baseline)", 'Abs Error': 0.0}

    for name, func in methods.items():
        val, exec_time = time_method_with_timeout(func, timeout, df=df_sample, x_val=1, z_cols=z_cols)
        error = abs(val - ground_truth) if isinstance(val, (int, float)) else "N/A"
        results[name] = {'Value': val, 'Time (s)': exec_time, 'Abs Error': error}
    
    # 3. Print results
    print(f"{'Method':<25} | {'Estimated':<18} | {'Acc':<18} | {'Time (s)':<20}")
    print("-" * 80)
    # Ensure a consistent order
    order = ["Naive Summation", "Empirical Summation", "Modeling (g-comp)"]
    for method in order:
        res = results[method]
        val_str = f"{res['Value']:.4f}" if isinstance(res['Value'], float) else str(res['Value'])
        err_str = f"{res['Abs Error']:.4f}" if isinstance(res['Abs Error'], float) else "N/A"
        time_str = f"{res['Time (s)']:.4f}" if isinstance(res['Time (s)'], float) else str(res['Time (s)'])
        print(f"{method:<25} | {val_str:<18} | {err_str:<18} | {time_str:<20}")

# --- Scenarios ---
# Using a large d to highlight the differences
run_comparison(d=2, n_samples=50000, timeout=3)