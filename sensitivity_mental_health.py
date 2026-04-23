"""
Sensitivity analysis: linear vs nonlinear mental health cost models.

Tests robustness of THEMIS conclusions to the functional form of the
mental health cost relationship with NPI stringency. Compares:
  1. Linear (baseline): depression cost proportional to adjust_factor
  2. Concave (saturation): depression cost proportional to adjust_factor^0.5
  3. Convex (threshold): depression cost proportional to adjust_factor^2

Since DELPHI predictions (cases, deaths) do not depend on the cost model,
we reuse existing rank-1 simulation results and recompute only the mental
health cost component under each functional form.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

from cost_functions.health_cost.health_data.health_params import MENTAL_HEALTH_COST
from cost_functions.economic_cost.economic_data.economic_params import TOTAL_GDP
from utils.visualization_utils import shorten_policy_string

REGIONS = ["DE", "US-NY", "ES", "BR"]
RESULTS_PATH = "simulation_results/rank1_scatter/combined_rank1_results.csv"
OUTPUT_DIR = "simulation_results/mh_sensitivity"

REGION_NAMES = {
    "US-NY": "New York, United States",
    "ES": "Spain",
    "DE": "Germany",
    "BR": "Brazil",
}

NONLINEAR_MODELS = {
    "concave": {"label": "Concave (saturation)", "func": np.sqrt, "alpha": 0.5},
    "convex":  {"label": "Convex (threshold)",   "func": lambda x: x**2, "alpha": 2.0},
}


def load_baseline_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    df["short_policy_name"] = [
        p if p == "actual" else shorten_policy_string(p)
        for p in df["policy"]
    ]
    return df


def add_nonlinear_mh_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Back-compute adjust_factor from stored mh_costs, then recompute
    mental health costs under each nonlinear model."""

    df["adjust_factor"] = 0.0
    for model_key in NONLINEAR_MODELS:
        for suffix in ["", "_lb", "_ub"]:
            df[f"mh_costs_{model_key}{suffix}"] = 0.0

    for region in REGIONS:
        mask = df["country"] == region
        MH = MENTAL_HEALTH_COST[region]

        dep_per_unit = (
            MH["gen_population_over14"]
            * MH["depression_rate_inc_gen_population"]
            * MH["depression_cost"]
        )
        ptsd_hw = (
            MH["exposed_health_workers"]
            * MH["ptsd_rate_inc_hworkers"]
            * MH["ptsd_cost"]
        )
        ptsd_sick_rate = MH["ptsd_rate_inc_sick"] * MH["ptsd_cost"]

        ptsd_point = ptsd_hw + df.loc[mask, "num_cases"] * ptsd_sick_rate
        dep_point  = df.loc[mask, "mh_costs"] - ptsd_point
        af = (dep_point / dep_per_unit).clip(lower=0)
        df.loc[mask, "adjust_factor"] = af

        cases_lb = df.loc[mask, "num_cases_lb"].fillna(df.loc[mask, "num_cases"])
        cases_ub = df.loc[mask, "num_cases_ub"].fillna(df.loc[mask, "num_cases"])

        for model_key, spec in NONLINEAR_MODELS.items():
            af_nl = spec["func"](af)
            dep_nl = dep_per_unit * af_nl
            df.loc[mask, f"mh_costs_{model_key}"]    = dep_nl + ptsd_hw + df.loc[mask, "num_cases"] * ptsd_sick_rate
            df.loc[mask, f"mh_costs_{model_key}_lb"] = dep_nl + ptsd_hw + cases_lb * ptsd_sick_rate
            df.loc[mask, f"mh_costs_{model_key}_ub"] = dep_nl + ptsd_hw + cases_ub * ptsd_sick_rate

    return df


def compute_derived_costs(df: pd.DataFrame, model_key: str = None):
    """Compute life_costs and total_cost columns for a given MH model."""
    suffix = f"_{model_key}" if model_key else ""

    mh      = df[f"mh_costs{suffix}"]
    mh_lb   = df[f"mh_costs{suffix}_lb"] if f"mh_costs{suffix}_lb" in df.columns else mh
    mh_ub   = df[f"mh_costs{suffix}_ub"] if f"mh_costs{suffix}_ub" in df.columns else mh

    life      = df["d_costs"]    + df["h_costs"]    + mh
    life_lb   = df["d_costs_lb"].fillna(df["d_costs"]) + df["h_costs_lb"].fillna(df["h_costs"]) + mh_lb
    life_ub   = df["d_costs_ub"].fillna(df["d_costs"]) + df["h_costs_ub"].fillna(df["h_costs"]) + mh_ub
    total     = life + df["st_economic_costs"]

    return life, life_lb, life_ub, total


def plot_functional_forms(output_dir: str):
    """Plot the three functional forms of f(x) used in the analysis."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = np.linspace(0, 0.30, 200)
    ax.plot(x, x, "k-", linewidth=2, label="Linear (baseline)")
    ax.plot(x, np.sqrt(x), "C0--", linewidth=2, label=r"Concave: $f(x) = x^{0.5}$ (saturation)")
    ax.plot(x, x**2, "C3-.", linewidth=2, label=r"Convex: $f(x) = x^{2}$ (threshold)")
    ax.set_xlabel("NPI stringency index (adjust factor)", fontsize=10)
    ax.set_ylabel(r"$f$(stringency) $\rightarrow$ depression scaling", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 0.30)
    ax.set_ylim(0, 0.60)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mh_functional_forms.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "mh_functional_forms.png"), dpi=200)
    plt.close(fig)
    print("  Saved mh_functional_forms.pdf")


def plot_scatter_comparison(df: pd.DataFrame, output_dir: str):
    """2x2 panel: each region shows scatter of linear vs concave humanitarian costs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, region in enumerate(REGIONS):
        ax = axes[i]
        rdf = df[df["country"] == region].copy()
        gdp = TOTAL_GDP[region]

        life_lin, _, _, _   = compute_derived_costs(rdf, model_key=None)
        life_con, _, _, _   = compute_derived_costs(rdf, model_key="concave")
        life_cvx, _, _, _   = compute_derived_costs(rdf, model_key="convex")

        econ = rdf["st_economic_costs"] / gdp * 100
        y_lin = life_lin / gdp * 100
        y_con = life_con / gdp * 100
        y_cvx = life_cvx / gdp * 100

        ax.scatter(econ, y_lin, s=12, alpha=0.5, color="black", label="Linear", zorder=3)
        ax.scatter(econ, y_con, s=12, alpha=0.5, color="C0", marker="^", label="Concave (saturation)", zorder=2)
        ax.scatter(econ, y_cvx, s=12, alpha=0.5, color="C3", marker="s", label="Convex (threshold)", zorder=1)

        ax.set_yscale("log")
        ax.set_xlabel("Economic Costs (% of GDP)", fontsize=10)
        ax.set_ylabel("Humanitarian Costs (% of GDP)", fontsize=10)
        ax.set_title(REGION_NAMES[region], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mh_sensitivity_scatter.pdf"), dpi=300)
    fig.savefig(os.path.join(output_dir, "mh_sensitivity_scatter.png"), dpi=200)
    plt.close(fig)
    print("  Saved mh_sensitivity_scatter.pdf")


def plot_best_policies_comparison(df: pd.DataFrame, output_dir: str, n: int = 15):
    """2x2 panel: top-n policies by total cost, stacked bars, linear vs nonlinear."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    cost_labels = {
        "st_economic_costs": "Economic",
        "d_costs": "Loss of Life",
        "h_costs": "Hospitalization",
        "mh_key": "Mental Health",
    }
    colors_map = {
        "st_economic_costs": "#636EFA",
        "d_costs": "#EF553B",
        "h_costs": "#00CC96",
        "mh_key": "#AB63FA",
    }

    for idx, region in enumerate(REGIONS):
        ax = axes[idx]
        rdf = df[df["country"] == region].copy()
        gdp = TOTAL_GDP[region]

        _, _, _, total_lin = compute_derived_costs(rdf, model_key=None)
        rdf["total_cost_lin"] = total_lin

        _, _, _, total_con = compute_derived_costs(rdf, model_key="concave")
        rdf["total_cost_con"] = total_con

        top_lin = rdf.nsmallest(n, "total_cost_lin")

        bar_width = 0.35
        x_pos = np.arange(n)

        for model_key, offset, alpha_val in [
            (None, -bar_width/2, 0.9),
            ("concave", bar_width/2, 0.6),
        ]:
            mh_col = "mh_costs" if model_key is None else f"mh_costs_{model_key}"
            bottom = np.zeros(n)
            model_label = "Linear" if model_key is None else "Concave"
            for cost_key, label in [("st_economic_costs", "Economic"),
                                     ("d_costs", "Loss of Life"),
                                     ("h_costs", "Hospitalization"),
                                     (mh_col, "Mental Health")]:
                vals = top_lin[cost_key].values / gdp * 100
                color = colors_map.get(cost_key, "#AB63FA")
                show_label = (idx == 0 and model_key is None)
                bar_label = label if show_label else None
                ax.bar(x_pos + offset, vals, bar_width, bottom=bottom,
                       color=color, alpha=alpha_val, label=bar_label,
                       edgecolor="white", linewidth=0.3)
                bottom += vals

        ax.set_xticks(x_pos)
        ax.set_xticklabels(top_lin["short_policy_name"].values, rotation=70,
                           fontsize=6, ha="right")
        ax.set_ylabel("Cost (% of GDP)", fontsize=9)
        ax.set_title(REGION_NAMES[region], fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].legend(fontsize=7, loc="upper right")

    fig.suptitle("Top Policies: Linear (solid) vs Concave (transparent) Mental Health Costs",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mh_sensitivity_best_policies.pdf"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "mh_sensitivity_best_policies.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved mh_sensitivity_best_policies.pdf")


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rank correlations and policy-ranking stability per region."""
    rows = []
    for region in REGIONS:
        rdf = df[df["country"] == region].copy()
        _, _, _, total_lin = compute_derived_costs(rdf, model_key=None)
        rdf["total_cost_lin"] = total_lin
        rank_lin = total_lin.rank()

        for model_key in NONLINEAR_MODELS:
            _, _, _, total_nl = compute_derived_costs(rdf, model_key=model_key)
            rdf[f"total_cost_{model_key}"] = total_nl
            rank_nl = total_nl.rank()

            rho, p_val = spearmanr(total_lin, total_nl)

            top10_lin = set(rdf.nsmallest(10, "total_cost_lin")["short_policy_name"])
            top10_nl  = set(rdf.nsmallest(10, f"total_cost_{model_key}")["short_policy_name"])
            overlap_10 = len(top10_lin & top10_nl)

            top5_lin = set(rdf.nsmallest(5, "total_cost_lin")["short_policy_name"])
            top5_nl  = set(rdf.nsmallest(5, f"total_cost_{model_key}")["short_policy_name"])
            overlap_5 = len(top5_lin & top5_nl)

            best_lin = rdf.loc[total_lin.idxmin(), "short_policy_name"]
            best_nl  = rdf.loc[total_nl.idxmin(), "short_policy_name"]

            pct_change_mh = (
                (rdf[f"mh_costs_{model_key}"] - rdf["mh_costs"]).abs()
                / rdf["mh_costs"].replace(0, np.nan)
            ).median() * 100

            pct_change_total = (
                (total_nl - total_lin).abs() / total_lin.replace(0, np.nan)
            ).median() * 100

            max_rank_disp = (rank_nl - rank_lin).abs().max()

            rows.append({
                "Region": region,
                "Model": NONLINEAR_MODELS[model_key]["label"],
                "Spearman rho": rho,
                "p-value": p_val,
                "Top-5 overlap": f"{overlap_5}/5",
                "Top-10 overlap": f"{overlap_10}/10",
                "Best policy (linear)": best_lin,
                "Best policy (nonlinear)": best_nl,
                "Same best?": best_lin == best_nl,
                "Median |%Delta MH|": f"{pct_change_mh:.1f}%",
                "Median |%Delta total|": f"{pct_change_total:.2f}%",
                "Max rank displacement": int(max_rank_disp),
            })

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading baseline rank-1 results ...")
    df = load_baseline_results()
    print(f"  {len(df)} scenarios loaded across {df['country'].nunique()} regions")

    print("Computing nonlinear mental health costs ...")
    df = add_nonlinear_mh_costs(df)

    print("\nGenerating figures ...")
    plot_functional_forms(OUTPUT_DIR)
    plot_scatter_comparison(df, OUTPUT_DIR)
    plot_best_policies_comparison(df, OUTPUT_DIR)

    print("\nSummary statistics:")
    summary = compute_summary_statistics(df)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "mh_sensitivity_summary.csv"), index=False)
    print(f"\n  Saved summary to {OUTPUT_DIR}/mh_sensitivity_summary.csv")

    print("\n--- Per-region detail ---")
    for region in REGIONS:
        rdf = df[df["country"] == region]
        gdp = TOTAL_GDP[region]
        mh_lin = rdf["mh_costs"].median()
        _, _, _, total_lin = compute_derived_costs(rdf, model_key=None)
        mh_frac = (mh_lin / total_lin.median()) * 100
        print(f"  {region}: median MH costs = {mh_lin/1e9:.2f}B "
              f"({mh_frac:.1f}% of total cost, "
              f"{mh_lin/gdp*100:.2f}% of GDP)")


if __name__ == "__main__":
    main()
