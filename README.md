# CausalPy

**CausalPy** is a research library for **causal-effect identification and estimation** in
*acyclic directed mixed graphs* (ADMGs) — directed graphs with latent confounders. Given a
causal graph, a treatment set `X`, and an outcome set `Y`, it can

1. **identify** whether the interventional distribution `P(Y | do(X))` is computable from
   observational data, and return the corresponding estimand (back-door / sequential
   back-door / front-door / Tian c-component); and
2. **estimate** that effect from data using cross-fitted **OM** (outcome model / g-formula),
   **IPW** (inverse-probability weighting), and **DML/AIPW** (doubly-robust) estimators.

A built-in structural-causal-model (SCM) layer lets you generate data with a known
ground truth, so you can benchmark the estimators end to end.

---

## Key concepts

### ADMGs and the naming convention
Graphs are `networkx.DiGraph`s. Latent confounders are represented as explicit nodes whose
names start with `U` (each `U` node points to the observed variables it confounds). The rest
of the library relies on a **strict naming convention** for variable roles:

| prefix | role |
|---|---|
| `X*` | treatment(s) |
| `Y*` | outcome(s) |
| `U*` | latent / unobserved confounder |
| anything else (e.g. `C`, `Z`, `M`, `L`) | observed covariate / mediator |

Multivariate (clustered) variables are columns sharing a base name (e.g. `C1, C2, …` for a
conceptual node `C`); a `cluster_map` ties the conceptual node to its columns.

### Identification vs. estimation
- **Identification** is symbolic: it decides *whether* and *how* `P(Y | do(X))` can be written
  as a functional of the observational distribution.
- **Estimation** is numeric: given that functional and a data frame, it computes the effect
  with machine-learning nuisances (XGBoost) and cross-fitting.

---

## Installation

CausalPy targets **Python 3.11**. Create an environment and install the pinned dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `networkx`, `scikit-learn`, `scipy`, `xgboost`,
`osqp` (balancing solver), `dill`, `tabulate`, `matplotlib`, `pyvis`. CausalPy is used as a
set of importable modules from the repository root (there is no `pip install causalpy`).

---

## Quickstart

Identify and estimate a back-door-adjusted effect, then check it against the SCM's ground
truth:

```python
import numpy as np
import example_SCM, identify, est_general, statmodules

# 1. A ground-truth SCM over an ADMG with a confounder C  ->  X -> Y  with  C -> X, C -> Y
scm, X, Y = example_SCM.BD_SCM(seednum=42, d=4)
G, X, Y = identify.preprocess_GXY_for_ID(scm.graph, X, Y)        # X=['X'], Y=['Y']

# 2. Identify P(Y | do(X)) symbolically
estimand = identify.causal_identification(G, X, Y, latex=False, copyTF=False)
print(estimand)
# -> P(Y | do(X)) = Σ_{c}P(Y | C, X) P(C)

# 3. Estimate it from 20k observational rows (OM / IPW / DML, cross-fitted)
obs = scm.generate_observational_samples(20000, seed=42)
ate = est_general.estimate_case_by_case(G, X, Y, np.array([1]), obs, cluster_variables=['C'])

# 4. Compare the outcome-model estimate to the ground truth P(Y=1 | do(X=x))
truth = statmodules.ground_truth(scm, X, Y, np.array([1]))
print("estimate (OM):", {k: round(float(v), 3) for k, v in ate['OM'].items()})
print("ground truth :", {k: round(float(v), 3) for k, v in truth.items()})
# estimate (OM): {(0,): 0.419, (1,): 0.583}
# ground truth : {(0,): 0.387, (1,): 0.574}
```

`ate` is keyed by estimator (`'OM'`, `'IPW'`, `'DML'`) and then by treatment value tuple
(`(0,)`, `(1,)`). The estimates *approximate* the ground truth (finite sample + estimator
variance); the plug-in **OM** estimate is typically the closest.

---

## How it works

`estimate_case_by_case(G, X, Y, y_val, obs_data, ...)` is the single estimation entry point.
It checks the graph against a sequence of identification criteria and dispatches to the
matching estimator:

```
back-door (BD) ─────────────► est_mSBD.estimate_BD
sequential back-door (mSBD) ─► est_mSBD.estimate_SBD   (single outcome)
                              └ est_mSBD.estimate_mSBD_y (multiple outcomes)
Tian / gTian / product ─────► est_general.estimate_{Tian, gTian, product_QD}
otherwise (general c-comp.) ─► est_general.estimate_general
```

Each estimator returns `(ATE, VAR, lower_CI, upper_CI)` dictionaries keyed by estimator and
treatment value, and uses K-fold cross-fitting with XGBoost nuisances and OSQP balancing
weights.

---

## Module map

| module | role |
|---|---|
| `identify.py` | identification algorithm — `causal_identification(G, X, Y)`, AC-tree construction |
| `graph.py` | ADMG utilities: c-components, parents/ancestors, topological order, cluster maps |
| `adjustment.py`, `mSBD.py`, `frontdoor.py`, `tian.py` | criteria: back-door, sequential back-door, front-door, Tian / c-component |
| `est_general.py` | estimation **dispatcher** `estimate_case_by_case`; c-component / Tian estimators |
| `est_mSBD.py` | back-door / sequential-back-door OM·IPW·DML cross-fitting estimators |
| `statmodules.py` | XGBoost nuisances, OSQP balancing, `ground_truth(...)`, performance metrics |
| `SCM.py` | `StructuralCausalModel` — build and sample ground-truth SCMs |
| `example_SCM.py` | benchmark ADMGs / SCMs (back-door, front-door, Napkin, Bhattacharya figures, …) |
| `random_generator.py` | random ADMG generation under identification constraints |
| `simulation.py` | Monte-Carlo simulation workflows over sample sizes |

### Public API at a glance

```python
identify.causal_identification(G, X, Y, latex=True, copyTF=True)            # -> estimand (str)
est_general.estimate_case_by_case(G, X, Y, y_val, obs_data, only_OM=False)  # -> ATE dict
statmodules.ground_truth(scm, X, Y, yval)                                   # -> {x_val: truth}

scm = SCM.StructuralCausalModel()
scm.add_unobserved_variable(name, distribution)
scm.add_observed_variable(name, equation, parents, noise_distribution)
scm.generate_samples(n, seed=...)                 # all variables
scm.generate_observational_samples(n, seed=...)   # observed (non-U) columns only
```

---

## Status and limitations

CausalPy is a research codebase. The estimation paths differ in maturity:

- **Well-tested:** back-door (`BD`), sequential back-door (`SBD`), and multi-outcome
  `mSBD` estimation with OM / IPW / DML. These are **deterministic across processes**
  (feature-column ordering is fixed).
- **Experimental:** the Tian / general **c-component** estimators run in **`only_OM`** mode;
  their IPW/DML composition is not implemented and raises `NotImplementedError`. Numeric
  correctness for continuous covariates / outcomes is a work in progress.
- Graphs that the AC-tree decomposition cannot handle raise a clear `NotImplementedError`
  instead of crashing.
- `estimate_SBD` is **single-outcome**; use `estimate_mSBD_y` for multiple outcomes (the
  dispatcher routes automatically by `len(Y)`).

---

## Tutorials

Worked, runnable tutorials live in [`tutorials/`](tutorials/) (Jupyter notebooks with a
Markdown copy of each):

1. **Identification** — define an ADMG and read off the estimand.
2. **Estimation** — generate data and estimate `P(Y | do(X))`, compared to ground truth.
3. **Custom SCM** — build your own structural causal model and sample from it.

---

## References

- R. Bhattacharya, R. Nabi, I. Shpitser. *Semiparametric Inference for Causal Effects in
  Graphical Models with Hidden Variables.* JMLR, 2022.
- J. Tian, J. Pearl. *A General Identification Condition for Causal Effects.* AAAI, 2002.
