# Tutorial 3 — Build your own structural causal model

The previous tutorials used built-in example graphs. Here we build a **structural
causal model (SCM)** from scratch with `SCM.StructuralCausalModel`, sample data from
it, and run the full identify → estimate → validate loop on our own model.

An SCM is defined by, for each variable: its **parents**, a **structural equation**
(how it is computed from its parents and noise), and a **noise distribution**.
CausalPy reads the parent lists to build the graph automatically.


```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import warnings; warnings.filterwarnings('ignore')
```

## Define the model

We build a confounded treatment effect: an observed confounder `C` drives both the
treatment `X` and the outcome `Y`. Every variable is binary; we squash a linear index
through the logistic function `expit` to turn it into a probability.

Each structural equation receives its parents as keyword arguments, plus `noise` (a
draw from the variable's noise distribution) and `num_sample`.


```python
import io, contextlib, numpy as np
from scipy import stats
from scipy.special import expit
import SCM, identify, est_general, statmodules

scm = SCM.StructuralCausalModel()


def f_C(noise, **kw):
    return np.random.binomial(1, 0.5, kw["num_sample"])             # C ~ Bernoulli(0.5)


def f_X(C, noise, **kw):
    return np.random.binomial(1, expit(0.8 * C + noise))           # X depends on C


def f_Y(C, X, noise, **kw):
    return np.random.binomial(1, expit(0.5 * C + 1.0 * X + noise))  # Y depends on C, X


scm.add_observed_variable("C", f_C, [],         stats.norm(0, 1))
scm.add_observed_variable("X", f_X, ["C"],      stats.norm(0, 1))
scm.add_observed_variable("Y", f_Y, ["C", "X"], stats.norm(0, 1))

print("nodes:", sorted(scm.graph.nodes()))
print("edges:", sorted(f"{a}->{b}" for a, b in scm.graph.edges()))
```

    nodes: ['C', 'X', 'Y']
    edges: ['C->X', 'C->Y', 'X->Y']


The graph `C -> X`, `C -> Y`, `X -> Y` was inferred from the parent lists — `C` is
a back-door confounder, exactly the structure from Tutorial 1.

> To add a **latent** confounder instead, use `scm.add_unobserved_variable("U_XY",
> dist)` and list `"U_XY"` as a parent of the variables it confounds. Unobserved
> `U...` nodes are dropped from `generate_observational_samples` but still used by the
> ground-truth engine.

## Sample data


```python
df = scm.generate_observational_samples(20000, seed=1)
print("sample:", df.shape, "columns:", list(df.columns))
print("means:", {c: round(float(df[c].mean()), 3) for c in df.columns})
print(df.head(3).to_string())
```

    sample: (20000, 3) columns: ['C', 'X', 'Y']
    means: {'C': 0.498, 'X': 0.583, 'Y': 0.66}
       C  X  Y
    0  0  1  0
    1  0  0  0
    2  0  0  1


## Identify and estimate on our own model

Because the model exposes a graph, the identification and estimation pipeline works on
it unchanged — and because it is a full SCM, we still have a ground truth to check
against.


```python
G, X, Y = identify.preprocess_GXY_for_ID(scm.graph, ["X"], ["Y"])

with contextlib.redirect_stdout(io.StringIO()):
    estimand = identify.causal_identification(G, X, Y, latex=False, copyTF=False)
    ate = est_general.estimate_case_by_case(
        G, X, Y, np.array([1]), df, cluster_variables=["C"])
    truth = statmodules.ground_truth(scm, X, Y, np.array([1]))

print("estimand:", estimand)
print("OM   :", {k: round(float(v), 3) for k, v in ate["OM"].items()})
print("truth:", {k: round(float(v), 3) for k, v in truth.items()})
```

    estimand: P(Y | do(X)) = Σ_{c}P(Y | C, X) P(C)
    OM   : {(0,): 0.557, (1,): 0.736}
    truth: {(0,): 0.55, (1,): 0.738}


The OM estimate tracks the ground truth closely: our hand-built model is identified
by the same back-door formula, estimated from data, and validated against its own known
effect.

---

That completes the tour — **identify** (Tutorial 1), **estimate** (Tutorial 2), and
**build your own model** (Tutorial 3). See the [README](../README.md) for the module
map and the full list of estimators and limitations.
