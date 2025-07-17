#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot simulation results with visible IQR ribbons, box‑plots, dominance
matrix, **and a relative‑efficiency fan chart**.

* All estimator labels "IPW" → "IW".
* Figure 4 (fan chart) visualises, for each estimator, RMSE ratios against
  a chosen baseline (default = "IW").
"""

import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import os 


# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------

def dict2df(perf_dict):
	"""Convert nested dict → long DataFrame."""
	rows = []
	for dgp_key, by_n in perf_dict.items():
		for n, by_est in by_n.items():
			for est, values in by_est.items():
				for run_id, val in enumerate(values, 1):
					rows.append((dgp_key, int(n), est, run_id, val))
	return pd.DataFrame(rows,
						columns=["SCM_seed", "num_sample", "estimator", "run", "acc"])

def read_performance_dict(folder: str, stem: str):
	with open(f"{folder}result_{stem}.pkl", "rb") as fh:
		return pickle.load(fh)

def dominance_matrix(cell_stats: pd.DataFrame, n: int) -> pd.DataFrame:
	"""Return %‑wins matrix at fixed *n* (lower metric = better)."""
	wide = cell_stats.query("num_sample == @n").pivot(index="SCM_seed",
													   columns="estimator",
													   values="mean")
	ests = wide.columns
	wins = pd.DataFrame(index=ests, columns=ests, dtype=float)
	for r in ests:
		for c in ests:
			wins.loc[r, c] = (wide[r] < wide[c]).mean() * 100
	return wins.round(1)

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
	# ---- simulation spec --------------------------------------------------
	seednum, simulation_round, scenario = 190602, 100, 4
	sim_date, sim_time = "250601", "0000"


	fontsize_xtick = 25
	fontsize_ytick = 25


	stem = (f"RandomSim_{sim_date}{sim_time}_seednum{seednum}_"
			f"scenario{scenario}_round{simulation_round}_numsim{simulation_round}")

	pkl_path = "log_experiments/pkl/"
	plot_output_dir = "log_experiments/plot/"
	# Ensure the output directory exists
	os.makedirs(plot_output_dir, exist_ok=True)
	output_filename = stem + ".png"
	output_filepath = os.path.join(plot_output_dir, output_filename)

	# ---- load + reshape ---------------------------------------------------
	perf_dict = read_performance_dict(pkl_path, stem)
	df = dict2df(perf_dict)
	df["estimator"].replace({"IPW": "IW"}, inplace=True)  # rename once

	# ---- helper aggregates ------------------------------------------------
	cell_stats = (df.groupby(["SCM_seed", "num_sample", "estimator"], as_index=False)
					["acc"].mean().rename(columns={"acc": "mean"}))

	summary = (df.groupby(["num_sample", "estimator"], as_index=False)
				 ["acc"].agg(median="median",
							   q25=lambda s: s.quantile(.25),
							   q75=lambda s: s.quantile(.75)))

	sns.set_style("whitegrid")
	# Define color_map, palette, and SHOW_LEGEND before they are used
	color_map = {
		"DML": "red",
		"OM": "blue",
		"IW": "green"
	}  # Initialize color_map with specified colors.
	palette = sns.color_palette()  # Initialize palette with a default seaborn palette
	SHOW_LEGEND = False # Set to True to show legend, False to hide

	# ---------- Figure 1 -------------------------------------------------- #
	fig1, ax1 = plt.subplots(figsize=(10, 8))
	plt.grid(False)
	for k, (est, g) in enumerate(summary.groupby("estimator", sort=False)):
		g = g.sort_values("num_sample")

		# --- use your explicit RGB choice when available -------------------
		c = color_map.get(est, palette[k])      #  ←  only change in this line

		ax1.plot(g["num_sample"], g["median"], marker="o", label=est, color=c)
		ax1.fill_between(g["num_sample"], g["q25"], g["q75"], alpha=.25, color=c)
		ax1.plot(g["num_sample"], g["q25"], ls="--", lw=.8, color=c, alpha=.7)
		ax1.plot(g["num_sample"], g["q75"], ls="--", lw=.8, color=c, alpha=.7)

	ax1.set_xscale("log")
	if (summary[["median", "q25", "q75"]] > 0).all().all():
		ax1.set_yscale("log")
	xticks = sorted(df["num_sample"].unique())

	ax1.set_xticks(xticks, labels=[str(t) for t in xticks], fontsize=fontsize_xtick)
	ax1.tick_params(axis='y', labelsize=fontsize_ytick)
	# ax1.set_title()

	# Apply a FuncFormatter to the y-axis for more readable tick labels
	# This will format log scale ticks (e.g., 0.1, 0.01) as decimal strings
	ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: str(y)))

	ax1.xaxis.set_major_formatter(
    FuncFormatter(lambda x, _:
                  '100' if x == 100 else
                  f'{int(x/1000)}k' if x >= 1000 else
                  str(int(x)))
	)
	if SHOW_LEGEND:
		ax1.legend(title="Estimator", bbox_to_anchor=(1.02, 1))
	fig1.tight_layout()

	# Save the figure before showing it
	fig1.savefig(output_filepath, bbox_inches='tight')
	print(f"Figure saved to {output_filepath}") # Optional: print confirmation
  
	# ---------------- show all -------------------------------------------
	plt.show()
