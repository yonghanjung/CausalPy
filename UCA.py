# UCA.py
# --------------------------------------------------------------

import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Set, Tuple, Sequence, Any, Callable
from pprint import pprint

import xgboost as xgb
import statmodules
import example_SCM
import identify 
import graph
import adjustment   

###############################################################################
# Utilities
###############################################################################
def _ensure_dot(v: str) -> str:
	"""Create a default controlled copy name if none supplied in dot_map."""
	return f"{v}dot"

def make_independent_copy(df: pd.DataFrame, base: str, copy_name: str, rng) -> None:
	"""Add an independent copy column by row-wise permutation."""
	df[copy_name] = df[base].sample(frac=1.0, replace=False, random_state=rng).values


def xgb_predict(model, data: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
	return model.predict(xgb.DMatrix(data[cols]))

###############################################################################
# Core routine
###############################################################################
def marginalize_kernel_pre(
	S: Set[str],
	Z_next: Set[str],
	S_Z_prev: Set[str],
	dot_map: Dict[str, str] | None = None,
) -> Tuple[List[str], List[str], List[str]]:
	"""
	Parameters
	----------
	S         symbols in μ^{i+1}(S)
	Z_next    symbols Z_{i+1}
	S_Z_prev  symbols S^{Z}_{i-1}
	dot_map   nature symbol → controlled copy  (e.g. {"X":"Xdot"})

	Returns
	-------
	S_eval      list[str]  (evaluation set)
	C_cond      list[str]  (conditioning set)
	after_steps list[str]  (actions  '', 'swap(A,B)', 'replace(Cdot->C)', 'subst(C->Cdot)')
	"""
	if dot_map is None:
		dot_map = {}
	inv_dot = {v_dot: V for V, v_dot in dot_map.items()}

	def dot_of(v: str) -> str:
		"""Return the controlled copy name, creating the default if unseen."""
		return dot_map.get(v, f"{v}dot")

	# ------------------------------------------------------------------
	# STEP 0 – build *candidate* set           (base vars only!)
	# ------------------------------------------------------------------
	base_candidates: Set[str] = set(S) | set(S_Z_prev)
	# if a controlled copy appears without its base, add the base **only to candidates**
	for v_dot in base_candidates.copy():
		if v_dot in inv_dot:
			base_candidates.add(inv_dot[v_dot])

	candidates = sorted(base_candidates - Z_next)

	# ------------------------------------------------------------------
	# Outputs
	# ------------------------------------------------------------------
	S_eval: List[str] = []
	C_cond: List[str] = []
	after_steps: List[str] = []

	# ------------------------------------------------------------------
	# iterate over base symbols (skip controlled names)
	# membership tests use the *original* S and S_Z_prev
	# ------------------------------------------------------------------
	for V in candidates:
		if V in dot_map.values():        # skip controlled copies themselves
			continue

		Vdot = dot_of(V)

		v_in_S      = V      in S
		v_in_SZ     = V      in S_Z_prev
		vdot_in_S   = Vdot   in S
		vdot_in_SZ  = Vdot   in S_Z_prev

		aft = ""     # default after-step is empty

		# ---------------- pattern dispatch ----------------------------
		if v_in_S and vdot_in_SZ:                         # P1
			S_eval.append(Vdot)
			C_cond.extend([V, Vdot])
			aft = f"swap({Vdot},{V})"

		elif vdot_in_S and vdot_in_SZ:                    # P2
			S_eval.append(V)
			C_cond.append(V)

		elif vdot_in_S and v_in_SZ:                       # P3
			S_eval.append(Vdot)
			C_cond.extend([Vdot, V])

		elif v_in_S and v_in_SZ:                          # P4
			S_eval.append(V)
			C_cond.append(V)

		elif v_in_S and not v_in_SZ:                      # P5
			if V not in dot_map:                          # create copy on the fly
				dot_map[V] = Vdot
			S_eval.append(Vdot)
			C_cond.append(Vdot)
			aft = f"replace({Vdot}->{V})"

		elif vdot_in_S and not vdot_in_SZ:                # P6
			S_eval.append(V)
			C_cond.append(V)

		elif v_in_SZ and not v_in_S:                      # P7
			C_cond.append(V)

		elif vdot_in_SZ and not vdot_in_S:                # P8
			C_cond.append(V)
			aft = f"subst({V}->{Vdot})"

		# --------------------------------------------------------------
		after_steps.append(aft)

	return S_eval, C_cond, after_steps


# ──────────────────────────────────────────────────────────────────────────
#  SECTION 3 ─  single-layer regression  μ̂ᶦ
# ──────────────────────────────────────────────────────────────────────────
def regress_one_stage(
    i: int,
    data_list: List[pd.DataFrame],
    kpp: List[Dict[str, Any]],
    dot_map: Dict[str, str],
    mu_next: Callable[[pd.DataFrame], np.ndarray] | None,
    outcome_col: str | None,
    rng_seed: int = 0,
    mu_params: Dict[str, Any] | None = None,
):
    """
    Learn μ̂ᶦ at stage i (1-based).

    Parameters
    ----------
    i            stage index (1 … m)
    data_list    [D1,…, D_{m+1}]
    kpp          KPP spec list (same length as data_list)
    dot_map      {base → copy}  e.g. {"X":"Xdot"}
    mu_next      μ̂^{i+1}  as a callable, or None if layer i+1 is Y
    outcome_col  name of Y in D_{i+1}  (required iff mu_next is None)
    rng_seed     reproducible permutation seed
    mu_params    XGBoost parameter override (or None for defaults)

    Returns
    -------
    model_i  fitted XGBoost booster for μ̂ᶦ
    info     dict  {S_eval, C_cond, after, cols}
             • cols = features used to predict with model_i
    """
    rng = np.random.default_rng(rng_seed)

    # ── 1. pull the two DataFrames Di, D_{i+1} ──────────────────────────
    spec_i, spec_ip1 = kpp[i - 1], kpp[i]
    D_i   = data_list[spec_i["index"]].copy()
    D_ip1 = data_list[spec_ip1["index"]].copy()

    # ── 2. symbolic sets for marginalisation ────────────────────────────
    S         = set(spec_ip1["cond_vars"]) | set(spec_ip1["nature_vars"])
    Z_next    = set(spec_ip1["nature_vars"])
    S_Z_prev  = set(spec_i["cond_vars"])

    S_eval, C_cond, after = marginalize_kernel_pre(
        S, Z_next, S_Z_prev, dot_map.copy()
    )

    # ── 3. make sure every needed “dot” copy exists in both frames ─────
    for df in (D_ip1, D_i):
        for col in S_eval + C_cond:
            if col.endswith("dot") and col not in df.columns:
                base = col[:-3]
                if base in df.columns:
                    make_independent_copy(df, base, col, rng)

    # ── 4. outcome vector y_target ──────────────────────────────────────
    if mu_next is None:
        if outcome_col is None:
            raise ValueError("Need outcome_col when mu_next is None.")
        y_target = D_ip1[outcome_col].to_numpy()
    else:
        # ensure D_ip1 has *all* columns μ^{i+1} will request
        next_cols = getattr(mu_next, "cols", None)
        if next_cols is None:
            # fallback: try to read from closure second cell
            try:
                next_cols = mu_next.__closure__[1].cell_contents
            except Exception:  # noqa: BLE001
                next_cols = []
        for col in next_cols:
            if col not in D_ip1.columns and col.endswith("dot"):
                base = col[:-3]
                if base in D_ip1.columns:
                    make_independent_copy(D_ip1, base, col, rng)
        # call μ^{i+1} on the *full* D_{i+1}
        y_target = mu_next(D_ip1)

    # ── 5. train μ̂ᶦ on (C_cond → y_target) ─────────────────────────────
    D_i["__y__"] = y_target
    model_i = statmodules.learn_mu(
        obs=D_i,
        col_feature=C_cond,
        col_label="__y__",
        params=mu_params,
    )

    # label the feature list so downstream λ can expose it via .cols
    model_i.cols = C_cond  # attach attribute for introspection

    return model_i, {"S_eval": S_eval, "C_cond": C_cond, "after": after, "cols": C_cond}


# --- Main Simulation Script ---
if __name__ == '__main__':

	num_sample = 10000
	seednum = 190602
	simulation_round = 10 
	list_seeds = list(np.random.randint(1,100000,size=simulation_round))

	scm, X, Y = example_SCM.Canonical_FD_SCM(seednum = seednum)   
	example_name = 'FD'
	cluster_variables = ['C']

	G = scm.graph
	G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
	topo_V = graph.find_topological_order(G)

	y_val = np.ones(len(Y)).astype(int)
	truth = statmodules.ground_truth(scm, X, Y, y_val)

	df_SCM = scm.generate_samples(num_sample, seed=list_seeds[0])
	observables = [node for node in df_SCM.columns if not node.startswith('U')]
	obs_data = df_SCM[observables]

	D1 = obs_data[["X", "C"]].copy()
	D2 = obs_data[["Z", "X", "C"]].copy()
	D3 = obs_data[["Y", "Z", "X", "C"]].copy()
	data_list = [D1, D2, D3]

	kpp = [
		dict(index=0, nature_vars=["X", "C"], cond_vars=[], policy_var=None, policy=None, surrogate_of=None),
		dict(index=1, nature_vars=["Z"],      cond_vars=["Xdot", "C"], policy_var="Xdot",
			 policy={"type": "do", "value": 1}, surrogate_of="X"),
		dict(index=2, nature_vars=["Y"],      cond_vars=["Z", "X", "C"], policy_var=None, policy=None,
			 surrogate_of=None)
	]
	# ---- backward recursion ---------------------------------------------
	m = len(kpp) - 1       # *** FIXED off-by-one ***
	mu_next, outcome = None, "Y"            # μ^{m+1} is just Y

	models = {}
	for i in range(m, 0, -1):               # i = m … 1
		mu_i, info = regress_one_stage(
			i, data_list, kpp, dot_map,
			mu_next=mu_next, outcome_col=outcome,
			rng_seed=123+i
		)
		models[i] = mu_i
		mu_next = lambda df, mdl=mu_i, cols=info["C_cond"]: xgb_predict(mdl, df, cols)
		outcome = None  # only the first iteration used raw Y



# --- This is just memo ---

	# print("Trained μ̂^1 and μ̂^2 without errors.")
	# dot_map = {"X": "Xdot"}

	# kpp_frontdoor = [
	# 	# P1  corresponds to D[0]
	# 	{
	# 	  "index":       0,
	# 	  "nature_vars": ["X", "C"],
	# 	  "cond_vars":   [],
	# 	  "policy_var":  None,
	# 	  "policy":      None,
	# 	  "surrogate_of": None
	# 	},
	# 	# P2  corresponds to D[1]
	# 	{
	# 	  "index":       1,
	# 	  "nature_vars": ["Z"],
	# 	  "cond_vars":   ["Xdot", "C"],          # uses the controlled Ẋ
	# 	  "policy_var":  "Xdot",
	# 	  "policy":      {"type":"do","value":1},
	# 	  "surrogate_of": "X"
	# 	},
	# 	# P3  corresponds to D[2]
	# 	{
	# 	  "index":       0,
	# 	  "nature_vars": ["Y"],
	# 	  "cond_vars":   ["Z", "X", "C"],
	# 	  "policy_var":  None,
	# 	  "policy":      None,
	# 	  "surrogate_of": None
	# 	}
	# ]

	# mu_next = None            # because layer i+1 is the outcome Y
	# outcome_col = "Y"         # column in D3
	# model_2, info_2 = regress_one_stage(
	# 	i            = 2,
	# 	data_list    = [obs_data],
	# 	kpp          = kpp_frontdoor,
	# 	dot_map      = {},
	# 	mu_next      = mu_next,
	# 	outcome_col  = outcome_col,
	# 	rng_seed     = 42,
	# )
	# print(info_2)






	# dot_map = {"X": "Xdot"}   # controlled counterpart

	# S = {"Z","X","C"}; Z_next = {"Z"}; S_Z_prev = {"Xdot"}
	# S = {"B","A","Xdot"}; Z_next = {"B"}; S_Z_prev = {"A","X"}
	# S = set(); Z_next = {"Y"}; S_Z_prev = {"Z","X","C"}
	# S = set(); Z_next = {"Y"}; S_Z_prev = {"B","A","Xdot"}
	# S = set(); Z_next = {"Y"}; S_Z_prev = {"B","A","Xdot"}

	# S_eval, C_cond, after_step_set = marginalize_kernel_pre(S, Z_next, S_Z_prev, dot_map)
	# pprint({
	#     "S_eval":      S_eval,
	#     "C_cond":      C_cond,
	#     "after_step":  after_step_set
	# })
	




