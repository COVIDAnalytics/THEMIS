"""
Ranking stability analysis for THEMIS under confounding perturbation.

For each region, we shrink all policy-specific gamma_{R,i} values toward
their cross-policy mean by a fraction f in [0,1], then re-simulate all
6^3 = 216 hypothetical policy sequences and record the full cost ranking.

At f=0, original estimates are used.  At f=1, all policies share an
identical gamma (i.e., NPIs have no differential effect).  This tests
how much of the estimated *differential* NPI effect must be attributable
to confounding before the policy ranking materially changes.

Metrics reported at each f:
  - Spearman rank correlation with baseline (f=0) ranking
  - Top-K overlap (K = 1, 5, 10, 20) with baseline top-K set
  - Identity of the cost-minimizing policy

Following the logic of Cinelli and Hazlett (2020), this framing directly
targets robustness of the *contrast* between policies rather than the
absolute level of gamma, avoiding artefacts in regions with low baseline
gamma (e.g., Germany).

Gamma estimation uses the full rank-1 ALS procedure from
analyze_gamma_rank.py, consistent with the updated methodology in the
paper (compare_rank1_vs_rank3.py).

Usage:
    python sensitivity_robustness_value.py --regions DE BR ES US-NY
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from analyze_gamma_rank import (
    POLICY_NAMES,
    POLICY_INDEX,
    build_gamma_matrix,
    rank1_imputation,
)
from pandemic_functions.pandemic import Pandemic_Factory, Pandemic
from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.pandemic_params import region_symbol_country_dict
from policy_functions.policy import Policy

FUTURE_POLICIES = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Restrict_Mass_Gatherings_and_Schools",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]

REGION_SYMBOL_TO_MATRIX_ID = {
    "DE": "Germany__None",
    "BR": "Brazil__None",
    "ES": "Spain__None",
    "US-NY": "US__New_York",
    "FR": "France__None",
    "US-FL": "US__Florida",
}


def matrix_row_to_gamma_dict(completed_row: np.ndarray) -> dict:
    """Convert a row of the completed gamma matrix to a policy->gamma dict."""
    return {p: float(completed_row[POLICY_INDEX[p]]) for p in POLICY_NAMES}


def blend_gammas(gamma_base: dict, f: float) -> dict:
    """
    Shrink each gamma toward the cross-policy mean by fraction f.
    At f=0: original values.  At f=1: all gammas equal to the mean.
    """
    mean_gamma = np.mean(list(gamma_base.values()))
    blended = {}
    for p in gamma_base:
        blended[p] = max(0.01, (1 - f) * gamma_base[p] + f * mean_gamma)
    return blended


def simulate_all_policies(factory, region, gamma_dict, start_date, policy_length):
    """Simulate all 6^policy_length policies and return cost DataFrame."""
    all_policies = list(itertools.product(FUTURE_POLICIES, repeat=policy_length))
    rows = []
    for pv in all_policies:
        label = "-".join(pv)
        try:
            policy = Policy(
                policy_type="hypothetical",
                start_date=start_date,
                policy_vector=list(pv),
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
            rows.append({
                "label": label,
                "economic": cost.st_economic_costs,
                "humanitarian": cost.d_costs + cost.h_costs + cost.mh_costs,
                "total": cost.st_economic_costs + cost.d_costs + cost.h_costs + cost.mh_costs,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def compute_ranking_stability(baseline_df, perturbed_df):
    """Compare rankings between baseline and perturbed cost DataFrames."""
    merged = baseline_df[["label", "total"]].merge(
        perturbed_df[["label", "total"]],
        on="label", suffixes=("_base", "_pert"),
    )
    if len(merged) < 10:
        return None

    rank_base = merged["total_base"].rank()
    rank_pert = merged["total_pert"].rank()

    rho, rho_p = spearmanr(rank_base, rank_pert)
    tau, tau_p = kendalltau(rank_base, rank_pert)

    n = len(merged)
    top_k_metrics = {}
    for k in [1, 5, 10, 20]:
        if k > n:
            continue
        base_topk = set(merged.nsmallest(k, "total_base")["label"])
        pert_topk = set(merged.nsmallest(k, "total_pert")["label"])
        overlap = len(base_topk & pert_topk)
        top_k_metrics[f"top{k}_overlap"] = overlap
        top_k_metrics[f"top{k}_overlap_pct"] = 100 * overlap / k

    opt_base = merged.loc[merged["total_base"].idxmin(), "label"]
    opt_pert = merged.loc[merged["total_pert"].idxmin(), "label"]

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "optimal_base": opt_base,
        "optimal_pert": opt_pert,
        "optimal_unchanged": opt_base == opt_pert,
        "n_policies": n,
        **top_k_metrics,
    }


def run_region(factory, region, base_gammas, start_date, policy_length, f_values):
    """Run full ranking stability sweep for a single region."""
    print(f"\n  Base gammas for {region} (rank-1 ALS):")
    for p, g in sorted(base_gammas.items()):
        print(f"    {p}: {g:.4f}")

    print(f"  Simulating {len(FUTURE_POLICIES)**policy_length} policies at f=0 (baseline) ...")
    baseline_df = simulate_all_policies(factory, region, base_gammas, start_date, policy_length)
    if len(baseline_df) == 0:
        print(f"  WARNING: No successful simulations for {region}")
        return None

    optimal_baseline = baseline_df.loc[baseline_df["total"].idxmin(), "label"]
    print(f"  Baseline optimal: {optimal_baseline}")
    print(f"  Successful simulations: {len(baseline_df)}/{len(FUTURE_POLICIES)**policy_length}")

    sweep_results = []
    for f in f_values:
        gamma_f = blend_gammas(base_gammas, f)
        print(f"    f={f:.2f} ...", end=" ")
        pert_df = simulate_all_policies(factory, region, gamma_f, start_date, policy_length)
        if len(pert_df) == 0:
            print("no simulations")
            continue

        metrics = compute_ranking_stability(baseline_df, pert_df)
        if metrics is None:
            print("insufficient overlap")
            continue

        metrics["f"] = float(f)
        sweep_results.append(metrics)
        print(f"rho={metrics['spearman_rho']:.3f}, top1={'same' if metrics['optimal_unchanged'] else metrics['optimal_pert']}")

    first_change_f = None
    for sr in sweep_results:
        if not sr["optimal_unchanged"]:
            first_change_f = sr["f"]
            break

    return {
        "region": region,
        "base_gammas": {k: round(v, 4) for k, v in sorted(base_gammas.items())},
        "optimal_baseline": optimal_baseline,
        "n_policies_simulated": len(baseline_df),
        "first_optimal_change_f": first_change_f,
        "sweep": sweep_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Ranking stability under confounding perturbation")
    parser.add_argument("--regions", nargs="+", default=["DE", "BR", "ES", "US-NY"])
    parser.add_argument("--startdate", default="2020-03-15")
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--f-step", type=float, default=0.05,
                        help="Step size for f sweep (default 0.05 = 21 points)")
    parser.add_argument("--output-dir", default="simulation_results/robustness_values")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    f_values = np.arange(0, 1 + args.f_step / 2, args.f_step)
    f_values = np.clip(f_values, 0, 1)

    print("=" * 70)
    print("  RANKING STABILITY ANALYSIS")
    print(f"  Sweep f in [0, 1] with step {args.f_step}")
    print(f"  f=0: original gammas.  f=1: all gammas equal (no NPI differentiation)")
    print(f"  Gamma estimation: rank-1 ALS (analyze_gamma_rank.py)")
    print("=" * 70)

    # Build gamma matrix and apply rank-1 ALS imputation (global)
    print("\n[0] Building gamma matrix and rank-1 ALS imputation ...")
    gamma_matrix, obs_mask, region_ids, _ = build_gamma_matrix(
        start_date="2020-03-15", end_date="2020-06-15", min_policy_days=10
    )
    rank1_completed, k_R = rank1_imputation(gamma_matrix, obs_mask)
    print(f"    {len(region_ids)} regions, {gamma_matrix.shape[1]} policies")

    region_id_to_idx = {r: i for i, r in enumerate(region_ids)}

    factory = Pandemic_Factory()
    all_results = {}

    for region in args.regions:
        print(f"\n{'=' * 60}")
        print(f"  Region: {region}")
        print(f"{'=' * 60}")

        matrix_id = REGION_SYMBOL_TO_MATRIX_ID.get(region)
        if matrix_id is None or matrix_id not in region_id_to_idx:
            print(f"  WARNING: {region} (matrix_id={matrix_id}) not found in gamma matrix, skipping")
            continue

        ri_idx = region_id_to_idx[matrix_id]
        base_gammas = matrix_row_to_gamma_dict(rank1_completed[ri_idx])

        rr = run_region(factory, region, base_gammas, args.startdate, args.length, f_values)
        if rr:
            all_results[region] = rr

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    summary_rows = []
    for region, rr in all_results.items():
        for sr in rr["sweep"]:
            summary_rows.append({
                "region": region,
                "f": sr["f"],
                "spearman_rho": sr["spearman_rho"],
                "kendall_tau": sr["kendall_tau"],
                "top1_unchanged": sr["optimal_unchanged"],
                "top5_overlap_pct": sr.get("top5_overlap_pct", None),
                "top10_overlap_pct": sr.get("top10_overlap_pct", None),
                "top20_overlap_pct": sr.get("top20_overlap_pct", None),
                "optimal_policy": sr["optimal_pert"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "ranking_stability.csv", index=False)

    for region, rr in all_results.items():
        print(f"\n  {region}:")
        print(f"    Baseline optimal: {rr['optimal_baseline']}")
        if rr["first_optimal_change_f"] is not None:
            print(f"    First optimal change at f = {rr['first_optimal_change_f']:.2f}")
        else:
            print(f"    Optimal policy NEVER changes across entire f sweep")

        f50 = [s for s in rr["sweep"] if abs(s["f"] - 0.50) < 0.01]
        if f50:
            s = f50[0]
            print(f"    At f=0.50: rho={s['spearman_rho']:.3f}, "
                  f"top5={s.get('top5_overlap_pct', 'N/A')}%, "
                  f"top20={s.get('top20_overlap_pct', 'N/A')}%")

    _plot_ranking_stability(all_results, output_dir)

    output_data = {}
    for region, rr in all_results.items():
        rr_slim = {k: v for k, v in rr.items() if k != "sweep"}
        rr_slim["sweep_summary"] = []
        for sr in rr["sweep"]:
            rr_slim["sweep_summary"].append({
                "f": sr["f"],
                "spearman_rho": sr["spearman_rho"],
                "kendall_tau": sr["kendall_tau"],
                "optimal_unchanged": sr["optimal_unchanged"],
                "optimal_pert": sr["optimal_pert"],
                "top1_overlap": sr.get("top1_overlap", None),
                "top5_overlap_pct": sr.get("top5_overlap_pct", None),
                "top10_overlap_pct": sr.get("top10_overlap_pct", None),
                "top20_overlap_pct": sr.get("top20_overlap_pct", None),
            })
        output_data[region] = rr_slim

    with open(output_dir / "full_results.json", "w") as out:
        json.dump(output_data, out, indent=2, default=str)

    print(f"\n  All outputs saved to: {output_dir}")
    print("=" * 70)


def _plot_ranking_stability(all_results, output_dir):
    """Plot Spearman rho and top-K overlap vs f for each region."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for region, rr in all_results.items():
        fs = [s["f"] for s in rr["sweep"]]
        rhos = [s["spearman_rho"] for s in rr["sweep"]]
        top20s = [s.get("top20_overlap_pct", 0) for s in rr["sweep"]]

        axes[0].plot(fs, rhos, "o-", label=region, markersize=4)
        axes[1].plot(fs, top20s, "s-", label=region, markersize=4)

    axes[0].axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="$\\rho = 0.8$")
    axes[0].set_xlabel("Confounding fraction $f$", fontsize=12)
    axes[0].set_ylabel("Spearman $\\rho$ (rank correlation)", fontsize=12)
    axes[0].set_title("Policy Ranking Stability", fontsize=13)
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(y=80, color="green", linestyle="--", alpha=0.5, label="80%")
    axes[1].set_xlabel("Confounding fraction $f$", fontsize=12)
    axes[1].set_ylabel("Top-20 overlap (%)", fontsize=12)
    axes[1].set_title("Top-20 Policy Set Stability", fontsize=13)
    axes[1].set_ylim(-5, 105)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "ranking_stability.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "ranking_stability.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
