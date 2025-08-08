import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper function for Bernoulli KL-divergence & its inversion ---

def kl_bernoulli(p, q):
    """Kullback-Leibler divergence for two Bernoulli distributions."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def solve_kl_ucb(p_hat, target_kl, search_range, n_iters=16):
    """Finds q by inverting the KL-divergence using binary search."""
    q_low, q_high = search_range
    for _ in range(n_iters):
        q_mid = (q_low + q_high) / 2.0
        kl = kl_bernoulli(p_hat, q_mid)
        # Search direction depends on whether we are finding upper or lower bound
        if search_range[0] < search_range[1]: # Searching for Upper Bound (q > p_hat)
            if kl < target_kl:
                q_low = q_mid
            else:
                q_high = q_mid
        else: # Searching for Lower Bound (q < p_hat)
            if kl < target_kl:
                q_high = q_mid
            else:
                q_low = q_mid
    return (q_low + q_high) / 2.0


# --- Algorithm Implementations ---

class KL_UCB:
    """Standard KL-UCB Algorithm (Baseline)."""
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.pull_counts = np.zeros(n_arms)
        self.reward_sums = np.zeros(n_arms)
        self.t = 0

    def select_arm(self):
        self.t += 1
        if self.t <= self.n_arms:
            return self.t - 1

        rhs = np.log(self.t) + 3 * np.log(np.log(self.t))
        ucb_indices = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            p_hat = self.reward_sums[arm] / self.pull_counts[arm]
            target_kl = rhs / self.pull_counts[arm]
            ucb_indices[arm] = solve_kl_ucb(p_hat, target_kl, (p_hat, 1.0))
        return np.argmax(ucb_indices)

    def update(self, arm, reward):
        self.pull_counts[arm] += 1
        self.reward_sums[arm] += reward

class Intersect_UCB:
    """
    The CORRECTED implementation of the 'shrinking by intersection' algorithm.
    """
    def __init__(self, n_arms, D_matrix):
        self.n_arms = n_arms
        self.D_matrix = D_matrix
        self.pull_counts = np.zeros(n_arms)
        self.reward_sums = np.zeros(n_arms)
        self.reward_sq_sums = np.zeros(n_arms)
        self.L_ett = np.zeros(n_arms)
        self.U_ett = np.ones(n_arms)
        self.t = 0

    def select_arm(self):
        self.t += 1
        if self.t <= self.n_arms:
            return self.t - 1

        final_upper_bounds = np.zeros(self.n_arms)
        rhs = np.log(self.t) + 3 * np.log(np.log(self.t))

        for arm in range(self.n_arms):
            p_hat = self.reward_sums[arm] / self.pull_counts[arm]
            target_kl = rhs / self.pull_counts[arm]
            U_emp = solve_kl_ucb(p_hat, target_kl, (p_hat, 1.0))
            U_ett_arm = self.U_ett[arm]
            final_upper_bounds[arm] = min(U_emp, U_ett_arm)

        return np.argmax(final_upper_bounds)

    def update(self, pulled_arm, reward):
        # Update empirical stats for the pulled arm
        self.pull_counts[pulled_arm] += 1
        self.reward_sums[pulled_arm] += reward
        self.reward_sq_sums[pulled_arm] += reward**2

        # ==================== THE FIX ====================
        # Calculate the FULL confidence interval for the pulled arm to propagate its uncertainty.
        rhs = np.log(self.t) + 3 * np.log(np.log(self.t))
        if self.pull_counts[pulled_arm] > 0:
            p_hat_pulled = self.reward_sums[pulled_arm] / self.pull_counts[pulled_arm]
            target_kl_pulled = rhs / self.pull_counts[pulled_arm]
            L_emp_pulled = solve_kl_ucb(p_hat_pulled, target_kl_pulled, (p_hat_pulled, 0.0))
            U_emp_pulled = solve_kl_ucb(p_hat_pulled, target_kl_pulled, (p_hat_pulled, 1.0))
        else: # Should not happen after initial phase
            L_emp_pulled, U_emp_pulled = 0, 1

        # Update ETT bounds for ALL OTHER arms based on the CI of the pulled arm
        for x in range(self.n_arms):
            if x == pulled_arm:
                continue

            # Estimate variance of arm x
            if self.pull_counts[x] > 1:
                mu_hat_x = self.reward_sums[x] / self.pull_counts[x]
                var_hat_x = (self.reward_sq_sums[x] / self.pull_counts[x]) - mu_hat_x**2
            else:
                var_hat_x = 0.25 # Max variance for Bernoulli
            var_hat_x = max(0, var_hat_x)

            delta = np.sqrt(2 * var_hat_x * self.D_matrix[x, pulled_arm])

            # New ETT interval is based on the PULLED ARM'S CI, not its point estimate.
            new_L_ett = L_emp_pulled - delta
            new_U_ett = U_emp_pulled + delta
            
            # Intersect this new constraint with the existing ETT bounds
            self.L_ett[x] = max(self.L_ett[x], new_L_ett)
            self.U_ett[x] = min(self.U_ett[x], new_U_ett)

# --- Simulation and Plotting --- (Identical to previous script)
def run_experiment(arm_means, D_matrix, horizon=3000, n_sims=200):
    n_arms = len(arm_means)
    mu_optimal = np.max(arm_means)
    
    regrets = {'KL-UCB': np.zeros((n_sims, horizon)), 'Corrected Intersect-UCB': np.zeros((n_sims, horizon))}
    pulls = {'KL-UCB': np.zeros((n_sims, n_arms)), 'Corrected Intersect-UCB': np.zeros((n_sims, n_arms))}
    algos = {'KL-UCB': KL_UCB(n_arms), 'Corrected Intersect-UCB': Intersect_UCB(n_arms, D_matrix)}

    print("Running simulations with corrected code...")
    for i in range(n_sims):
        if (i + 1) % 20 == 0: print(f"  Simulation {i+1}/{n_sims}")
        for name, algo_class in algos.items():
            current_algo = algo_class.__class__(n_arms, D_matrix) if name != 'KL-UCB' else algo_class.__class__(n_arms)
            sim_rewards = []
            for t in range(horizon):
                arm = current_algo.select_arm()
                reward = np.random.binomial(1, arm_means[arm])
                current_algo.update(arm, reward)
                sim_rewards.append(reward)
            pulls[name][i, :] = current_algo.pull_counts
            regrets[name][i, :] = mu_optimal * np.arange(1, horizon + 1) - np.cumsum(sim_rewards)
    print("Simulations complete.")
    return regrets, pulls

# Main Execution
ARM_MEANS = np.array([0.8, 0.75, 0.5])
N_ARMS = len(ARM_MEANS)
D_MATRIX = np.full((N_ARMS, N_ARMS), 0.5); D_MATRIX[0, 1] = 0.01; D_MATRIX[1, 0] = 0.01; np.fill_diagonal(D_MATRIX, 0)
regrets, pulls = run_experiment(ARM_MEANS, D_MATRIX)

# Plotting
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(np.mean(regrets['KL-UCB'], axis=0), label="KL-UCB")
plt.plot(np.mean(regrets['Corrected Intersect-UCB'], axis=0), label="Corrected Intersect-UCB")
plt.title("Cumulative Regret (Corrected Algorithm)")
plt.xlabel("Time Steps (T)"); plt.ylabel("Average Cumulative Regret"); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
bar_width = 0.35; index = np.arange(N_ARMS)
plt.bar(index - bar_width/2, np.mean(pulls['KL-UCB'], axis=0), bar_width, label="KL-UCB")
plt.bar(index + bar_width/2, np.mean(pulls['Corrected Intersect-UCB'], axis=0), bar_width, label="Corrected Intersect-UCB")
plt.title("Average Arm Pulls"); plt.xlabel("Arm Index"); plt.ylabel("Total Pulls")
plt.xticks(index, [f"Arm {i}\n($\mu$={m})" for i,m in enumerate(ARM_MEANS)]); plt.legend(); plt.tight_layout(); plt.show()