# Simulation comparing three estimators for continuous-dose CATE:
# 1) Orthogonal "basis" R-learner (Pattern 1: fixed dose basis + centered regressors + ridge + anchor)
# 2) Orthogonal "black-box" R-learner (Pattern 2: stop-gradient centering with alternation + anchor)
# 3) Non-orthogonal naive learner (regress Y - m(C) on (C,X) without centering; anchor)
#
# We inject n^{-1/4}-rate noise into nuisance estimates m(C) and the centering pieces
# to stress-test orthogonality. We evaluate RMSE of tau_hat(c,x) on a test distribution.
#
# NOTE: This is a compact demonstration (not production code); emphasis is clarity over speed.

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from numpy.linalg import solve
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(42)

# ---------- Data Generating Process (DGP) ----------

def generate_data(n, seed=0):
    rng_local = np.random.default_rng(seed)
    # Covariates: 2D Gaussian
    C = rng_local.normal(size=(n, 2))
    c1, c2 = C[:,0], C[:,1]
    # Treatment assignment mechanism
    def e_func(C):
        c1, c2 = C[:,0], C[:,1]
        return 0.6*c1 - 0.5*c2 + 0.3*c1*c2
    e = e_func(C)
    X = e + rng_local.normal(scale=1.0, size=n)  # sigma_x = 1
    # Outcome regression m(x,c)
    def m_func(x, C):
        c1, c2 = C[:,0], C[:,1]
        f_c = 0.8 + 0.5*np.sin(c1) + 0.25*(c2**2)
        a_c = 0.5 + 0.3*c1
        b_c = 0.15 + 0.05*c2
        q_c = 0.3/(1.0 + np.exp(-(c1 + 0.5*c2)))
        return f_c + a_c*x + b_c*(x**2) + q_c*np.sin(x + 0.5*c1)
    eps_y = rng_local.normal(scale=0.6, size=n)  # sigma_y = 0.6
    Y = m_func(X, C) + eps_y
    # True tau(c,x) anchored at x0=0
    def tau_true(c, x, x0=0.0):
        return m_func(x, c.reshape(-1,2)) - m_func(np.full_like(x, x0), c.reshape(-1,2))
    return C, X, Y, e_func, m_func, tau_true

# ---------- Utilities ----------

def make_phi(x):
    # Dose basis phi(x): [x, x^2, x^3, x^4, sin x, cos x]
    return np.column_stack([x, x**2, x**3, x**4, np.sin(x), np.cos(x)])

def psi_features(C, degree=3):
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    return poly.fit_transform(C), poly  # returns features and the transformer for test use

def center_phi_given_C(phi_X, C, kfold, noise_sd):
    # For each coordinate j of phi(X), fit E[phi_j(X)|C] via cross-fitting
    n, J = phi_X.shape
    Gamma_hat = np.zeros((n, J))
    # use GradientBoostingRegressor per coordinate
    for j in range(J):
        # Cross-fitting
        for train_idx, test_idx in kfold.split(C):
            gbr = GradientBoostingRegressor(random_state=123, max_depth=3, n_estimators=150, learning_rate=0.05)
            gbr.fit(C[train_idx], phi_X[train_idx, j])
            pred = gbr.predict(C[test_idx])
            Gamma_hat[test_idx, j] = pred
    # Inject n^{-1/4}-rate noise (per-sample)
    n_root_4 = n**(-0.25)
    Gamma_hat_noisy = Gamma_hat + rng.normal(scale=noise_sd * n_root_4, size=Gamma_hat.shape)
    return Gamma_hat_noisy

def crossfit_m_hat(C, Y, kfold, noise_sd):
    n = len(Y)
    m_hat = np.zeros(n)
    for train_idx, test_idx in kfold.split(C):
        gbr = GradientBoostingRegressor(random_state=202, max_depth=3, n_estimators=200, learning_rate=0.05)
        gbr.fit(C[train_idx], Y[train_idx])
        m_hat[test_idx] = gbr.predict(C[test_idx])
    # Add n^{-1/4} noise
    m_hat_noisy = m_hat + rng.normal(scale=noise_sd * n**(-0.25), size=n)
    return m_hat_noisy

def basis_method(C, X, Y, x0=0.0, rho=0.01, noise_sd_m=0.5, noise_sd_phi=0.5):
    n = len(Y)
    kfold = KFold(n_splits=2, shuffle=True, random_state=7)
    # Step 1: m_hat(C) cross-fitted + noise
    m_hat = crossfit_m_hat(C, Y, kfold, noise_sd=noise_sd_m)
    Y_perp = Y - m_hat
    # Step 2: phi(X) and centered regressors Z = phi(X) - Gamma_hat(C), with cross-fitted Gamma + noise
    phi_X = make_phi(X)  # n x J
    Gamma_hat = center_phi_given_C(phi_X, C, kfold, noise_sd=noise_sd_phi)  # actually produces Gamma_hat (not centered)
    Z = phi_X - Gamma_hat  # centered in dose
    # Step 3: psi(C) basis
    psi_C, psi_transformer = psi_features(C, degree=3)  # n x K
    # Build Psi and centered tildePsi via tensor product
    # Psi_i = kron(phi_X[i], psi_C[i])
    # tildePsi_i = kron(Z[i], psi_C[i])
    J = phi_X.shape[1]
    K = psi_C.shape[1]
    # Construct matrices efficiently
    Psi = np.einsum('ij,ik->ijk', phi_X, psi_C).reshape(n, J*K)
    tPsi = np.einsum('ij,ik->ijk', Z, psi_C).reshape(n, J*K)
    # Ridge closed-form solution
    R = (tPsi.T @ tPsi) / n
    Q = (Psi.T @ Psi) / n
    b = (tPsi.T @ Y_perp) / n
    # Small additional numerical ridge to avoid singularity
    lam_eye = 1e-6 * np.eye(J*K)
    beta_hat = solve(R + rho*Q + lam_eye, b)
    # Define prediction functions
    def h_hat(c, x):
        psi_test = psi_transformer.transform(c.reshape(1, -1))  # 1 x K
        phi_test = make_phi(np.array([x]))  # 1 x J
        psi_kron = np.kron(phi_test, psi_test).reshape(J*K)
        return float(psi_kron @ beta_hat)
    # Vectorized prediction for arrays
    def h_hat_vec(Ctest, Xtest):
        psi_test = psi_transformer.transform(Ctest)  # M x K
        phi_test = make_phi(Xtest)  # M x J
        Psi_test = np.einsum('ij,ik->ijk', phi_test, psi_test).reshape(len(Xtest), J*K)
        return Psi_test @ beta_hat
    return h_hat, h_hat_vec, rho

def blackbox_pattern2(C, X, Y, x0=0.0, rho=0.0, noise_sd_m=0.5, noise_sd_r=0.5, n_alternations=2, random_state=1):
    # Cross-fit m_hat first
    n = len(Y)
    kfold = KFold(n_splits=2, shuffle=True, random_state=11)
    m_hat = crossfit_m_hat(C, Y, kfold, noise_sd=noise_sd_m)
    Y_perp = Y - m_hat
    # Prepare folds
    folds = list(kfold.split(C))
    models_h = []
    scalers = []
    # Initialize r predictions to zero for both folds
    r_pred = [np.zeros(len(idx)) for _, idx in folds]  # per-fold r(C) on that fold
    # Alternate
    for alt in range(n_alternations):
        new_r_models = []
        # For each fold: train h with r fixed (from the other fold)
        for f_idx, (train_idx, hold_idx) in enumerate(folds):
            # Features for h-step on hold-out fold
            X_hold = np.column_stack([C[hold_idx], X[hold_idx]])
            T_hold = Y_perp[hold_idx] + r_pred[f_idx]  # target = Y_perp + r(C) (r fixed)
            # Scale features (helps NN)
            scaler = StandardScaler().fit(X_hold)
            Xh = scaler.transform(X_hold)
            # Train MLP
            h_model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', alpha=1e-4,
                                   learning_rate_init=0.01, learning_rate='adaptive',
                                   max_iter=300, early_stopping=True, n_iter_no_change=10,
                                   random_state=random_state + alt*10 + f_idx)
            h_model.fit(Xh, T_hold)
            # Store
            if alt == n_alternations - 1:
                models_h.append(h_model)
                scalers.append(scaler)
            # Predict h on the *other* fold (train_idx) to fit r there (independent split)
            X_other = np.column_stack([C[train_idx], X[train_idx]])
            Xo = scaler.transform(X_other)
            h_pred_other = h_model.predict(Xo)
            # Fit r(C) on the other fold
            r_model = GradientBoostingRegressor(random_state=99 + f_idx + alt*3, max_depth=3, n_estimators=150, learning_rate=0.05)
            r_model.fit(C[train_idx], h_pred_other)
            new_r_models.append(r_model)
        # After training r on each other fold, produce new r_pred for each fold
        for f_idx, (train_idx, hold_idx) in enumerate(folds):
            r_model_otherfold = new_r_models[1 - f_idx]  # use model fitted on the opposite split for independence
            # Predict r on this fold
            r_on_hold = r_model_otherfold.predict(C[hold_idx])
            # Inject n^{-1/4} noise
            r_on_hold_noisy = r_on_hold + rng.normal(scale=noise_sd_r * (n**(-0.25)), size=len(hold_idx))
            r_pred[f_idx] = r_on_hold_noisy
    # Define predictors: average the two fold-specific h models for any new (c,x)
    def h_hat(c, x):
        feats = np.array([[c[0], c[1], x]])
        preds = []
        for model, scaler in zip(models_h, scalers):
            preds.append(float(model.predict(scaler.transform(feats))))
        return float(np.mean(preds))
    def h_hat_vec(Ctest, Xtest):
        feats = np.column_stack([Ctest, Xtest])
        agg = np.zeros(len(Xtest))
        for model, scaler in zip(models_h, scalers):
            agg += model.predict(scaler.transform(feats))
        return agg / len(models_h)
    return h_hat, h_hat_vec, rho

def naive_non_orthogonal(C, X, Y, x0=0.0, noise_sd_m=0.5, random_state=3):
    # Cross-fit m_hat, add n^{-1/4} noise, then regress Y - m_hat(C) on (C,X) without centering
    n = len(Y)
    kfold = KFold(n_splits=2, shuffle=True, random_state=21)
    m_hat = crossfit_m_hat(C, Y, kfold, noise_sd=noise_sd_m)
    Y_perp = Y - m_hat
    # Fit two fold-specific h models on their held-out folds (to mimic the same data usage style)
    models_h = []
    scalers = []
    for train_idx, hold_idx in kfold.split(C):
        X_hold = np.column_stack([C[hold_idx], X[hold_idx]])
        T_hold = Y_perp[hold_idx]
        scaler = StandardScaler().fit(X_hold)
        Xh = scaler.transform(X_hold)
        h_model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', alpha=1e-4,
                               learning_rate_init=0.01, learning_rate='adaptive',
                               max_iter=300, early_stopping=True, n_iter_no_change=10,
                               random_state=random_state)
        h_model.fit(Xh, T_hold)
        models_h.append(h_model)
        scalers.append(scaler)
    def h_hat(c, x):
        feats = np.array([[c[0], c[1], x]])
        preds = []
        for model, scaler in zip(models_h, scalers):
            preds.append(float(model.predict(scaler.transform(feats))))
        return float(np.mean(preds))
    def h_hat_vec(Ctest, Xtest):
        feats = np.column_stack([Ctest, Xtest])
        agg = np.zeros(len(Xtest))
        for model, scaler in zip(models_h, scalers):
            agg += model.predict(scaler.transform(feats))
        return agg / len(models_h)
    return h_hat, h_hat_vec

# ---------- Evaluation ----------

def evaluate_methods(n, R=3, x0=0.0, seed_base=1000):
    rows = []
    for r in range(R):
        seed = seed_base + r
        C, X, Y, e_func, m_func, tau_true = generate_data(n, seed=seed)
        # Fit basis method
        h_b, h_b_vec, rho_b = basis_method(C, X, Y, x0=x0, rho=0.01, noise_sd_m=0.5, noise_sd_phi=0.5)
        # Fit black-box pattern 2
        h_bb, h_bb_vec, rho_bb = blackbox_pattern2(C, X, Y, x0=x0, rho=0.0, noise_sd_m=0.5, noise_sd_r=0.5,
                                                   n_alternations=2, random_state=seed)
        # Fit naive method
        h_nv, h_nv_vec = naive_non_orthogonal(C, X, Y, x0=x0, noise_sd_m=0.5, random_state=seed+1)
        # Test set drawn from the same support to respect positivity
        n_test = 4000
        C_te, X_te, _, _, m_func_te, tau_true_te = generate_data(n_test, seed=seed+9999)
        x0_arr = np.full(n_test, x0)
        tau_true_vals = tau_true_te(C_te, X_te, x0_arr)
        # Compute predictions and anchor
        # Basis
        h_b_pred = h_b_vec(C_te, X_te)
        h_b_pred_x0 = h_b_vec(C_te, x0_arr)
        tau_hat_b = (1.0 + rho_b)*(h_b_pred - h_b_pred_x0)
        # Black-box
        h_bb_pred = h_bb_vec(C_te, X_te)
        h_bb_pred_x0 = h_bb_vec(C_te, x0_arr)
        tau_hat_bb = (1.0 + rho_bb)*(h_bb_pred - h_bb_pred_x0)
        # Naive
        h_nv_pred = h_nv_vec(C_te, X_te)
        h_nv_pred_x0 = h_nv_vec(C_te, x0_arr)
        tau_hat_nv = h_nv_pred - h_nv_pred_x0
        # RMSE
        rmse_b = np.sqrt(mean_squared_error(tau_true_vals, tau_hat_b))
        rmse_bb = np.sqrt(mean_squared_error(tau_true_vals, tau_hat_bb))
        rmse_nv = np.sqrt(mean_squared_error(tau_true_vals, tau_hat_nv))
        rows.append({"n": n, "rep": r+1, "method": "Orthogonal - Basis", "RMSE": rmse_b})
        rows.append({"n": n, "rep": r+1, "method": "Orthogonal - BlackBox", "RMSE": rmse_bb})
        rows.append({"n": n, "rep": r+1, "method": "Naive (Non-orthogonal)", "RMSE": rmse_nv})
    df = pd.DataFrame(rows)
    summary = df.groupby(["n","method"]).agg(RMSE_mean=("RMSE","mean"),
                                             RMSE_std=("RMSE","std")).reset_index()
    return df, summary

ns = [500, 1000, 5000, 10000]
all_rows = []
all_summary = []
for n in ns:
    df_n, summ_n = evaluate_methods(n, R=20, x0=0.0, seed_base=1000 + n)
    all_rows.append(df_n)
    all_summary.append(summ_n)

df_results = pd.concat(all_rows, ignore_index=True)
df_summary = pd.concat(all_summary, ignore_index=True)

# import caas_jupyter_tools
# caas_jupyter_tools.display_dataframe_to_user("Simulation results (per replicate)", df_results)

# Plot RMSE vs n (log scale on y for readability)
import matplotlib.pyplot as plt

plt.figure()
for method in df_summary["method"].unique():
    sub = df_summary[df_summary["method"]==method].sort_values("n")
    plt.plot(sub["n"], sub["RMSE_mean"], marker="o", label=method)
plt.xscale("linear")
plt.yscale("log")
plt.xlabel("Sample size n")
plt.ylabel("RMSE (log scale)")
plt.title("CATE RMSE vs sample size (with n^{-1/4} nuisance noise)")
plt.legend()
plt.tight_layout()
plt.show()
