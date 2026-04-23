"""
Sensitivity analysis for unmeasured confounding in THEMIS.

Introduces a multiplicative bias factor delta into gamma_{R,i} and re-runs
the THEMIS pipeline across a grid of delta values to assess how robust the
efficient frontier and optimal policy rankings are to plausible confounding.

If voluntary behavior inflates the estimated NPI effect, the "true" gamma
would be gamma_{R,i}^true = gamma_{R,i}^obs * delta, where delta > 1 means
NPIs appear less effective (gamma closer to 1 = no measure).

Usage:
    python sensitivity_confounding_sweep.py --region DE
    python sensitivity_confounding_sweep.py --region DE --deltas 0.8 0.9 1.0 1.1 1.2 1.3 1.5
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandemic_functions.pandemic import Pandemic_Factory, Pandemic
from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.pandemic_params import region_symbol_country_dict
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    get_region_gammas_v2,
)
from policy_functions.policy import Policy


FUTURE_POLICIES = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Restrict_Mass_Gatherings_and_Schools",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]

DEFAULT_DELTAS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]

COST_COLS = [
    "policy", "st_economic_costs", "d_costs", "h_costs", "mh_costs",
    "num_deaths", "num_cases", "humanitarian_costs", "economic_costs", "total_costs",
]


def perturb_gammas(gamma_dict: dict, delta: float) -> dict:
    """
    Apply confounding bias factor delta to gamma values.

    delta > 1 means NPIs are less effective than estimated (gamma moves toward 1).
    delta < 1 means NPIs are more effective than estimated (gamma moves toward 0).

    We blend toward the no-measure baseline (gamma=1):
        gamma_perturbed = 1 - (1 - gamma_obs) / delta
    This ensures No_Measure stays at 1.0 and the reduction (1-gamma) shrinks by 1/delta.
    """
    perturbed = {}
    for policy, gamma in gamma_dict.items():
        reduction = 1.0 - gamma
        perturbed_reduction = reduction / delta
        perturbed[policy] = max(0.01, min(2.0, 1.0 - perturbed_reduction))
    return perturbed


def simulate_single_policy(args):
    """Simulate a single policy vector and return cost dict."""
    factory, region, policy_vect, gamma_dict, start_date = args
    try:
        policy = Policy(
            policy_type="hypothetical",
            start_date=start_date,
            policy_vector=policy_vect,
        )
        country, province = region_symbol_country_dict[region]
        country_sub = country.replace(" ", "_")
        province_sub = province.replace(" ", "_")
        totalcases = pd.read_csv(
            f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
        )
        pandemic = Pandemic(
            policy, region, factory.delphi_prediction,
            totalcases, gamma_dict
        )
        cost = PandemicCost(pandemic)
        label = "-".join(policy_vect)

        econ = cost.st_economic_costs
        humanitarian = cost.d_costs + cost.h_costs + cost.mh_costs
        total = econ + humanitarian

        return {
            "policy": label,
            "policy_vector": policy_vect,
            "st_economic_costs": econ,
            "d_costs": cost.d_costs,
            "h_costs": cost.h_costs,
            "mh_costs": cost.mh_costs,
            "num_deaths": cost.num_deaths,
            "num_cases": cost.num_cases,
            "humanitarian_costs": humanitarian,
            "economic_costs": econ,
            "total_costs": total,
        }
    except Exception as e:
        return {
            "policy": "-".join(policy_vect),
            "policy_vector": policy_vect,
            "error": str(e),
        }


def run_sweep_for_delta(factory, region, delta, start_date, policy_length):
    """Run full policy grid for a single delta value."""
    base_gammas, _, _ = get_region_gammas_v2(region)
    perturbed = perturb_gammas(base_gammas, delta)

    scenarios = list(itertools.product(FUTURE_POLICIES, repeat=policy_length))
    results = []

    for policy_vect in scenarios:
        result = simulate_single_policy(
            (factory, region, list(policy_vect), perturbed, start_date)
        )
        if "error" not in result:
            results.append(result)

    return results


def compute_pareto_frontier(df, econ_col="economic_costs", human_col="humanitarian_costs"):
    """Compute Pareto-optimal policies (minimize both economic and humanitarian)."""
    pareto = []
    sorted_df = df.sort_values(econ_col).reset_index(drop=True)
    min_human = float("inf")
    for _, row in sorted_df.iterrows():
        if row[human_col] < min_human:
            pareto.append(row)
            min_human = row[human_col]
    return pd.DataFrame(pareto)


def rank_policies_by_total_cost(df, top_k=20):
    """Return the top-K policies by total cost."""
    return df.nsmallest(top_k, "total_costs")["policy"].tolist()


def main():
    parser = argparse.ArgumentParser(description="Confounding sensitivity sweep")
    parser.add_argument("--region", "-r", type=str, required=True)
    parser.add_argument("--startdate", "-sd", type=str, default="2020-03-15")
    parser.add_argument("--length", "-l", type=int, default=3)
    parser.add_argument("--deltas", nargs="+", type=float, default=None)
    parser.add_argument("--output-dir", default="simulation_results/confounding_sensitivity")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    region = args.region
    start_date = args.startdate
    policy_length = args.length
    deltas = args.deltas if args.deltas else DEFAULT_DELTAS
    top_k = args.top_k
    output_dir = Path(args.output_dir) / region
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  CONFOUNDING SENSITIVITY SWEEP: {region}")
    print(f"  Deltas: {deltas}")
    print("=" * 70)

    factory = Pandemic_Factory()

    base_gammas, _, _ = get_region_gammas_v2(region)
    print(f"\n  Base gamma values for {region}:")
    for p, g in sorted(base_gammas.items()):
        print(f"    {p}: {g:.4f}")

    all_results = {}
    pareto_sets = {}
    top_k_rankings = {}

    for delta in deltas:
        print(f"\n--- Delta = {delta:.2f} ---")
        perturbed = perturb_gammas(base_gammas, delta)
        print("  Perturbed gammas:")
        for p, g in sorted(perturbed.items()):
            print(f"    {p}: {g:.4f}")

        results = run_sweep_for_delta(factory, region, delta, start_date, policy_length)
        df = pd.DataFrame(results)

        if len(df) == 0:
            print("  WARNING: No successful simulations")
            continue

        df.to_csv(output_dir / f"sweep_delta_{delta:.2f}.csv", index=False)
        all_results[delta] = df

        pareto = compute_pareto_frontier(df)
        pareto_sets[delta] = pareto
        print(f"  Pareto-optimal policies ({len(pareto)}):")
        for _, row in pareto.head(5).iterrows():
            print(f"    {row['policy']}: econ={row['economic_costs']:.2e}, human={row['humanitarian_costs']:.2e}")

        top_k_list = rank_policies_by_total_cost(df, top_k)
        top_k_rankings[delta] = top_k_list
        print(f"  Top-{top_k} by total cost (first 5): {top_k_list[:5]}")

    if len(all_results) < 2:
        print("Not enough successful delta runs for comparison.")
        return

    # --- Analysis: stability of rankings ---
    print("\n" + "=" * 70)
    print("  STABILITY ANALYSIS")
    print("=" * 70)

    baseline_df = all_results.get(1.0)
    if baseline_df is None:
        baseline_delta = min(deltas, key=lambda d: abs(d - 1.0))
        baseline_df = all_results[baseline_delta]
        print(f"  Using delta={baseline_delta} as baseline (1.0 not in grid)")
    else:
        baseline_delta = 1.0

    baseline_top_k = rank_policies_by_total_cost(baseline_df, top_k)
    baseline_top1 = baseline_top_k[0]

    stability_report = []
    for delta in sorted(all_results.keys()):
        df = all_results[delta]
        this_top_k = rank_policies_by_total_cost(df, top_k)
        overlap = len(set(baseline_top_k) & set(this_top_k))
        top1_match = this_top_k[0] == baseline_top1

        baseline_total = baseline_df.set_index("policy")["total_costs"]
        delta_total = df.set_index("policy")["total_costs"]
        common = baseline_total.index.intersection(delta_total.index)
        rank_corr = baseline_total[common].rank().corr(delta_total[common].rank(), method="spearman")

        stability_report.append({
            "delta": delta,
            "top1_policy": this_top_k[0],
            "top1_matches_baseline": top1_match,
            f"top{top_k}_overlap_with_baseline": overlap,
            f"top{top_k}_overlap_pct": 100 * overlap / top_k,
            "spearman_rank_corr": rank_corr,
        })
        print(f"  delta={delta:.2f}: top1={this_top_k[0]}, "
              f"overlap={overlap}/{top_k}, rho={rank_corr:.4f}")

    stability_df = pd.DataFrame(stability_report)
    stability_df.to_csv(output_dir / "ranking_stability.csv", index=False)

    # --- Pareto frontier comparison plot ---
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=min(deltas), vmax=max(deltas))

    for delta in sorted(all_results.keys()):
        df = all_results[delta]
        color = cmap(norm(delta))
        lw = 3 if abs(delta - 1.0) < 0.01 else 1.5
        alpha = 1.0 if abs(delta - 1.0) < 0.01 else 0.7
        ax.scatter(
            df["economic_costs"], df["humanitarian_costs"],
            c=[color], s=8, alpha=0.3,
        )
        pareto = pareto_sets.get(delta)
        if pareto is not None and len(pareto) > 1:
            pareto_sorted = pareto.sort_values("economic_costs")
            ax.plot(
                pareto_sorted["economic_costs"],
                pareto_sorted["humanitarian_costs"],
                color=color, linewidth=lw, alpha=alpha,
                label=f"$\\delta$={delta:.2f}",
            )

    ax.set_xlabel("Economic Costs", fontsize=12)
    ax.set_ylabel("Humanitarian Costs", fontsize=12)
    ax.set_title(f"Efficient Frontier Under Confounding Bias ({region})", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(output_dir / "pareto_frontier_sensitivity.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "pareto_frontier_sensitivity.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # --- Ranking stability plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    deltas_sorted = sorted(stability_df["delta"])

    axes[0].plot(deltas_sorted, stability_df.set_index("delta").loc[deltas_sorted, f"top{top_k}_overlap_pct"],
                 "o-", color="#1b9e77", linewidth=2, markersize=6)
    axes[0].axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Confounding Factor $\\delta$", fontsize=11)
    axes[0].set_ylabel(f"Top-{top_k} Overlap with Baseline (%)", fontsize=11)
    axes[0].set_title("Policy Ranking Stability", fontsize=12)
    axes[0].set_ylim(0, 105)

    axes[1].plot(deltas_sorted, stability_df.set_index("delta").loc[deltas_sorted, "spearman_rank_corr"],
                 "o-", color="#d95f02", linewidth=2, markersize=6)
    axes[1].axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Confounding Factor $\\delta$", fontsize=11)
    axes[1].set_ylabel("Spearman Rank Correlation", fontsize=11)
    axes[1].set_title("Full Ranking Correlation", fontsize=12)
    axes[1].set_ylim(0, 1.05)

    top1_match = stability_df.set_index("delta").loc[deltas_sorted, "top1_matches_baseline"].astype(int)
    axes[2].bar(deltas_sorted, top1_match, width=0.03, color="#7570b3")
    axes[2].axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Confounding Factor $\\delta$", fontsize=11)
    axes[2].set_ylabel("Top-1 Matches Baseline", fontsize=11)
    axes[2].set_title("Optimal Policy Stability", fontsize=12)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["No", "Yes"])

    plt.tight_layout()
    fig.savefig(output_dir / "ranking_stability.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "ranking_stability.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # --- Summary ---
    summary = {
        "region": region,
        "start_date": start_date,
        "policy_length": policy_length,
        "deltas_tested": deltas,
        "baseline_delta": baseline_delta,
        "baseline_top1": baseline_top1,
        "baseline_top_k": baseline_top_k,
        "base_gammas": {k: round(v, 4) for k, v in base_gammas.items()},
        "stability": stability_report,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  All outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
