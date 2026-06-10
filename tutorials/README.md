# CausalPy tutorials

Three worked, runnable notebooks. Each has a rendered Markdown copy next to it so you
can read it on GitHub without launching Jupyter.

1. **[Identification](01_identification.ipynb)** &nbsp;([markdown](01_identification.md))
   — turn a causal graph into an estimand across seven cases: back-door, front-door,
   the Napkin graph, a nested Napkin, a multi-treatment sequential (mSBD) effect,
   plan identification, and a non-identifiable bow — each with the **drawn ADMG**
   and the **rendered LaTeX formula**.
2. **[Estimation](02_estimation.ipynb)** &nbsp;([markdown](02_estimation.md))
   — estimate effects on **four graphs**: back-door, a multi-treatment sequential
   regime `do(X1, X2)`, the Napkin ratio-form c-component effect (all with
   OM / balancing-IPW / DML), and a one-dimensional front-door graph (general
   c-component, OM-only) — each with the drawn ADMG and a comparison to the SCM
   ground truth.
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
