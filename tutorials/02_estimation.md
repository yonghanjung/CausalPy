# Tutorial 2 — Estimation: from an estimand to a number

[Tutorial 1](01_identification.ipynb) turned a graph into an *estimand*. This one
estimates that estimand from data and checks it against a known ground truth.

The single entry point is

```python
est_general.estimate_case_by_case(G, X, Y, y_val, obs_data, cluster_variables=[...])
```

It inspects the graph, picks the matching estimator, and returns estimates from three
methods: **OM** (outcome-model / g-formula plug-in), **IPW** (inverse-probability
weighting), and **DML** (doubly-robust). All three use K-fold cross-fitting with
gradient-boosted nuisances.


```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import warnings; warnings.filterwarnings('ignore')
```

## A back-door effect with known ground truth

`example_SCM.BD_SCM` builds a back-door graph — the same `C -> X -> Y`, `C -> Y`
pattern as Tutorial 1 — together with the structural causal model behind it, so we can
both sample data and compute the true `P(Y=1 | do(X=x))`. Here `d=4` makes the
confounder **multivariate**: it appears as four continuous columns `C1, …, C4`, and
`cluster_variables=["C"]` tells the estimator they form one conceptual node `C`.


```python
import io, contextlib, numpy as np
import example_SCM, identify, est_general, statmodules

scm, X, Y = example_SCM.BD_SCM(seednum=42, d=4)
G, X, Y = identify.preprocess_GXY_for_ID(scm.graph, X, Y)

obs = scm.generate_observational_samples(20000, seed=42)
print("observational sample:", obs.shape, "columns:", list(obs.columns))
print(obs.head(3).to_string())
```

    observational sample: (20000, 6) columns: ['C1', 'C2', 'C3', 'C4', 'X', 'Y']
             C1        C2        C3        C4  X  Y
    0  0.163648 -0.187795 -0.631953  0.068962  0  0
    1  0.734806  0.676857  0.115722  1.746774  1  1
    2  0.778183  0.920495 -0.293999 -1.037220  1  0


## Estimate `P(Y | do(X))`

We estimate the effect at `X = 1` (the function returns the effect at every treatment
level it enumerates) and compare each estimator to the ground truth.


```python
with contextlib.redirect_stdout(io.StringIO()):
    ate = est_general.estimate_case_by_case(
        G, X, Y, np.array([1]), obs, cluster_variables=["C"])
    truth = statmodules.ground_truth(scm, X, Y, np.array([1]))


def show(name, d):
    print(f"{name:6s}", {k: round(float(v), 3) for k, v in d.items()})


for est in ["OM", "IPW", "DML"]:
    show(est, ate[est])
show("truth", truth)
```

    OM     {(0,): 0.419, (1,): 0.583}
    IPW    {(0,): 0.471, (1,): 0.6}
    DML    {(0,): 0.471, (1,): 0.6}
    truth  {(0,): 0.387, (1,): 0.574}


Read the keys as treatment values: `(0,)` is `do(X=0)`, `(1,)` is `do(X=1)`. All
three estimators land near the ground truth; the plug-in **OM** is closest here. The
gaps are finite-sample and estimator variance, and shrink with more data.

| | `do(X=0)` | `do(X=1)` |
|---|---|---|
| ground truth | ≈ 0.39 | ≈ 0.57 |
| OM | ≈ 0.42 | ≈ 0.58 |
| IPW / DML | ≈ 0.47 | ≈ 0.60 |


## What the three estimators are

- **OM** — fit `E[Y | C, X]`, then average over `P(C)` (the g-formula plug-in).
- **IPW** — reweight observed outcomes by balancing weights so the treated and
  untreated covariate distributions match.
- **DML** — combine both into a doubly-robust estimate (consistent if *either* the
  outcome model or the weights are right), with cross-fitting to remove overfitting bias.

For **back-door** and **sequential-back-door** graphs all three are available. For the
**c-component / Tian** estimators (such as the Napkin effect from Tutorial 1) only the
**OM** path is implemented today — see the README's *Status and limitations*.

---

**Next:** [Tutorial 3 — Custom SCM](03_custom_scm.ipynb) builds a structural causal
model from scratch and runs this same identify → estimate → validate loop on it.
