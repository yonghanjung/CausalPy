# Tutorial 1 — Identification: from a causal graph to an estimand

**Identification** asks a symbolic question: given a causal graph with latent
confounders, *can* the interventional distribution `P(Y | do(X))` be written as a
formula in the observational distribution — and if so, *which* formula?

CausalPy answers this with a single call:

```python
identify.causal_identification(G, X, Y)
```

It returns the estimand as a human-readable string, or `None` when the effect is
**not identifiable**. This notebook walks through four cases: back-door, front-door,
a c-component ("Napkin") effect, and a non-identifiable one.

> Graphs are `networkx.DiGraph`s. Latent confounders are explicit nodes named
> `U...` (e.g. `U_XY` points to both `X` and `Y`). Treatments are named `X...`,
> outcomes `Y...`, and everything else is an observed covariate.


```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import warnings; warnings.filterwarnings('ignore')
```

## A quiet helper

`causal_identification` prints its reasoning as it works. To keep the output tidy we
wrap it so we only see the returned estimand, plus a helper that builds the graph for
a named example SCM.


```python
import io, contextlib, networkx as nx
import example_SCM, identify


def idq(G, X, Y):
    """Run identification quietly; return just the estimand (or None)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return identify.causal_identification(G, X, Y, latex=False, copyTF=False)


def graph_of(name):
    """Build the (preprocessed) graph, treatment and outcome of an example SCM."""
    scm, X, Y = getattr(example_SCM, name)(seednum=42)
    return identify.preprocess_GXY_for_ID(scm.graph, X, Y)
```

## 1. Back-door

The simplest case: a single **observed** confounder `C` that affects both the
treatment `X` and the outcome `Y`.


```python
G, X, Y = graph_of("BD_SCM")
print("treatment:", X, " outcome:", Y)
print("edges:", sorted(f"{a}->{b}" for a, b in G.edges()))
print("estimand:", idq(G, X, Y))
```

    treatment: ['X']  outcome: ['Y']
    edges: ['C->X', 'C->Y', 'X->Y']
    estimand: P(Y | do(X)) = Σ_{c}P(Y | C, X) P(C)


`C` opens a *back-door* path `X <- C -> Y`. Because `C` is observed, adjusting for
it blocks the path and gives the classic adjustment formula

`P(Y | do(X)) = Σ_c P(Y | C, X) P(C)`.

> The left-to-right order *within* a conditioning set (e.g. `P(Y | C, X)` vs
> `P(Y | X, C)`) comes from Python set iteration and can differ between runs — it is the
> same set and the same estimand.

## 2. Front-door

Now the graph is richer: a mediator `Z` carries the effect (`X -> Z -> Y`), while a
**latent** confounder `U_XY` sits between `X` and `Y` and *cannot* be adjusted for.
The effect is still identifiable — through `Z`.


```python
G, X, Y = graph_of("Canonical_FD_SCM")
print("edges:", sorted(f"{a}->{b}" for a, b in G.edges()))
print("estimand:", idq(G, X, Y))
```

    edges: ['C->X', 'C->Y', 'C->Z', 'U_CX->C', 'U_CX->X', 'U_CY->C', 'U_CY->Y', 'U_XY->X', 'U_XY->Y', 'X->Z', 'Z->Y']
    estimand: P(Y | do(X)) = Σ_{z, c} P(Z | C, X)P(C) Σ_{x} P(Y | Z, C, X)P(X | C)


Even though `U_XY -> X` and `U_XY -> Y` are unobservable, the mediator `Z` is
unconfounded with `X`, so CausalPy identifies the effect as a two-stage,
front-door-style expression.

## 3. A c-component effect (the "Napkin" graph)

Some effects are identifiable even though *neither* back-door nor front-door applies.
The **Napkin** graph is the classic example; CausalPy identifies it through Tian's
c-component factorization, which yields a **ratio** estimand.


```python
G, X, Y = graph_of("Napkin_SCM")
print("edges:", sorted(f"{a}->{b}" for a, b in G.edges()))
print("estimand:", idq(G, X, Y))
```

    edges: ['R->X', 'U_WX->W', 'U_WX->X', 'U_WY->W', 'U_WY->Y', 'W->R', 'X->Y']
    estimand: P({Y} | do({X})) = [[Σ_{w}P(X, Y | R, W) P(W)]/[Σ_{w}P(X | R, W) P(W)]]


The answer is a ratio of marginalized conditionals — not a single adjustment sum.
CausalPy derives it automatically from the graph's c-component structure.

## 4. When it is *not* identifiable

If a treatment and outcome are joined by **both** a direct edge `X -> Y` *and* a
latent confounder `U_XY` (a "bow"), no formula in the observational distribution
equals `P(Y | do(X))`. Identification returns `None` — and, run unquietly, it explains
why.


```python
Gb = nx.DiGraph()
Gb.add_edges_from([("U_XY", "X"), ("U_XY", "Y"), ("X", "Y")])
Gb, Xb, Yb = identify.preprocess_GXY_for_ID(Gb, ["X"], ["Y"])

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    result = identify.causal_identification(Gb, Xb, Yb, latex=False, copyTF=False)

print(buf.getvalue().strip())     # the function's own explanation
print("returned:", result)
```

    P(Y | do(X)) is not identifiable from G, since Q[['Y']] is not identifiable from G(['X', 'Y'])
    returned: None


A `None` return is CausalPy's verdict that the effect cannot be identified from
observational data alone.

---

**Next:** [Tutorial 2 — Estimation](02_estimation.ipynb) takes an identifiable effect
and estimates it from data, checking the answer against a known ground truth.
