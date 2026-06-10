# CausalPy tutorials

Three worked, runnable notebooks. Each has a rendered Markdown copy next to it so you
can read it on GitHub without launching Jupyter.

1. **[Identification](01_identification.ipynb)** &nbsp;([markdown](01_identification.md))
   — turn a causal graph into an estimand: back-door, front-door, a c-component
   ("Napkin") effect, and a non-identifiable case.
2. **[Estimation](02_estimation.ipynb)** &nbsp;([markdown](02_estimation.md))
   — estimate `P(Y | do(X))` from data with the OM / IPW / DML estimators and compare
   to a known ground truth.
3. **[Custom SCM](03_custom_scm.ipynb)** &nbsp;([markdown](03_custom_scm.md))
   — build your own structural causal model from scratch and run the full
   identify → estimate → validate loop on it.

## Running them

From the repository root, in the environment that has `requirements.txt` installed:

```bash
pip install jupyter      # for the notebooks only — not a library dependency
jupyter lab tutorials/
```

Each notebook's first cell adds the repository root to `sys.path`, so imports such as
`import example_SCM` resolve when the notebook is launched from `tutorials/`. The
Markdown copies are generated from the executed notebooks with `jupyter nbconvert --to
markdown`.
