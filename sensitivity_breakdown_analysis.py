"""
Breakdown analysis: compute the minimum confounding factor delta* that
reverses key policy comparisons from THEMIS.

For each focal policy pair (e.g., optimal vs. lockdown, optimal vs. no-action),
find the smallest delta such that the cost ordering flips. This directly
answers "how wrong would our gamma estimates need to be for Conclusion X
to change?"

Also provides calibrated context by comparing delta* to the plausible
range of confounding from the empirical literature on voluntary behavior
change vs. mandated NPI effects.

Usage:
    python sensitivity_breakdown_analysis.py --region DE
    python sensitivity_breakdown_analysis.py --region DE --region BR --region ES --region US-NY
"""
import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

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
from sensitivity_confounding_sweep import perturb_gammas, FUTURE_POLICIES


# ---------------------------------------------------------------------------
# Literature-calibrated plausible range
# ---------------------------------------------------------------------------

LITERATURE_CALIBRATION = {
    "upper_bound_delta": 1.40,
    "central_estimate_delta": 1.20,
    "lower_bound_delta": 1.05,
    "sources": [
        {
            "authors": "Maloney and Taskin",
            "year": 2020,
            "journal": "NBER Working Paper",
            "finding": "Voluntary mobility reductions preceded mandated lockdowns by 1-2 weeks; "
                       "Google mobility declined ~25-35% before government orders.",
            "implied_delta_range": "1.10-1.35",
        },
        {
            "authors": "Abouk and Heydari",
            "year": 2021,
            "journal": "Health Economics",
            "finding": "Social distancing increased 5-10pp before stay-at-home orders; "
                       "orders added 5-12pp additional distancing.",
            "implied_delta_range": "1.05-1.25",
        },
        {
            "authors": "Yan et al.",
            "year": 2021,
            "journal": "PNAS",
            "finding": "Fear-driven voluntary behavior accounted for 45% of total "
                       "transmission reduction during stay-at-home orders.",
            "implied_delta_range": "1.20-1.45",
        },
        {
            "authors": "Gupta et al.",
            "year": 2021,
            "journal": "Journal of Public Economics",
            "finding": "Stay-at-home orders reduced mobility by only 5-10% beyond "
                       "the pre-existing voluntary trend.",
            "implied_delta_range": "1.05-1.15",
        },
    ],
    "interpretation": (
        "The literature suggests that voluntary behavior changes inflated the "
        "apparent effect of NPIs by approximately 5-45%, with most estimates "
        "in the 10-25% range. This corresponds to a plausible delta range of "
        "approximately 1.05 to 1.40, with a central estimate around 1.20."
    ),
}


# ---------------------------------------------------------------------------
# Core simulation for a single policy under a given delta
# ---------------------------------------------------------------------------

def simulate_policy_cost(factory, region, policy_vector, gamma_dict, start_date="2020-03-15"):
    """Run a single policy and return total, economic, humanitarian costs."""
    policy = Policy(
        policy_type="hypothetical",
        start_date=start_date,
        policy_vector=policy_vector,
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
    econ = cost.st_economic_costs
    humanitarian = cost.d_costs + cost.h_costs + cost.mh_costs
    return {
        "economic": econ,
        "humanitarian": humanitarian,
        "total": econ + humanitarian,
        "d_costs": cost.d_costs,
        "h_costs": cost.h_costs,
        "mh_costs": cost.mh_costs,
        "num_deaths": cost.num_deaths,
    }


# ---------------------------------------------------------------------------
# Find the optimal policy at delta=1
# ---------------------------------------------------------------------------

def find_optimal_and_key_policies(factory, region, base_gammas, start_date, policy_length):
    """Identify the optimal policy and key comparison policies."""
    all_policies = list(itertools.product(FUTURE_POLICIES, repeat=policy_length))
    results = []
    for pv in all_policies:
        pv_list = list(pv)
        try:
            costs = simulate_policy_cost(factory, region, pv_list, base_gammas, start_date)
            results.append({"policy_vector": pv_list, **costs})
        except Exception:
            continue

    if not results:
        return None

    df = pd.DataFrame(results)
    df["policy_label"] = df["policy_vector"].apply(lambda x: "-".join(x))
    optimal_idx = df["total"].idxmin()
    optimal = df.loc[optimal_idx]

    lockdown_label = "-".join(["Lockdown"] * policy_length)
    no_action_label = "-".join(["No_Measure"] * policy_length)

    key_comparisons = [
        ("optimal_vs_lockdown", optimal["policy_label"], lockdown_label),
        ("optimal_vs_no_action", optimal["policy_label"], no_action_label),
    ]

    pareto = _compute_pareto(df)
    if len(pareto) > 1:
        pareto_top = pareto.iloc[0]["policy_label"]
        pareto_second = pareto.iloc[1]["policy_label"]
        if pareto_top != optimal["policy_label"]:
            key_comparisons.append(("pareto_top_vs_second", pareto_top, pareto_second))

    return {
        "optimal": optimal.to_dict(),
        "key_comparisons": key_comparisons,
        "all_results": df,
    }


def _compute_pareto(df):
    """Simple Pareto frontier (minimize both economic and humanitarian)."""
    pareto = []
    sorted_df = df.sort_values("economic").reset_index(drop=True)
    min_human = float("inf")
    for _, row in sorted_df.iterrows():
        if row["humanitarian"] < min_human:
            pareto.append(row)
            min_human = row["humanitarian"]
    return pd.DataFrame(pareto)


# ---------------------------------------------------------------------------
# Bisection search for breakdown delta*
# ---------------------------------------------------------------------------

def find_breakdown_delta(
    factory, region, base_gammas,
    policy_a_vec, policy_b_vec,
    cost_key="total",
    start_date="2020-03-15",
    delta_range=(0.5, 3.0),
    tol=0.01,
    max_iter=30,
) -> dict:
    """
    Find the minimum delta such that cost(A) > cost(B), when at delta=1
    cost(A) < cost(B) (A is better).

    Returns the breakdown delta* and associated information.
    """
    cost_a_base = simulate_policy_cost(factory, region, policy_a_vec, base_gammas, start_date)
    cost_b_base = simulate_policy_cost(factory, region, policy_b_vec, base_gammas, start_date)

    if cost_a_base[cost_key] >= cost_b_base[cost_key]:
        return {
            "status": "already_reversed",
            "delta_star": 1.0,
            "cost_a_base": cost_a_base[cost_key],
            "cost_b_base": cost_b_base[cost_key],
            "margin_at_base": cost_b_base[cost_key] - cost_a_base[cost_key],
        }

    lo, hi = delta_range
    sign_at_lo = _cost_diff(factory, region, base_gammas, policy_a_vec, policy_b_vec, lo, cost_key, start_date)
    sign_at_hi = _cost_diff(factory, region, base_gammas, policy_a_vec, policy_b_vec, hi, cost_key, start_date)

    if sign_at_hi <= 0:
        return {
            "status": "not_found_in_range",
            "delta_star": float("inf"),
            "cost_a_base": cost_a_base[cost_key],
            "cost_b_base": cost_b_base[cost_key],
            "margin_at_base": cost_b_base[cost_key] - cost_a_base[cost_key],
            "delta_range_tested": list(delta_range),
        }

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        diff = _cost_diff(factory, region, base_gammas, policy_a_vec, policy_b_vec, mid, cost_key, start_date)
        if abs(diff) < tol * abs(cost_a_base[cost_key]):
            break
        if diff > 0:
            hi = mid
        else:
            lo = mid

    delta_star = (lo + hi) / 2

    return {
        "status": "found",
        "delta_star": float(delta_star),
        "cost_a_base": cost_a_base[cost_key],
        "cost_b_base": cost_b_base[cost_key],
        "margin_at_base": cost_b_base[cost_key] - cost_a_base[cost_key],
        "margin_pct": 100 * (cost_b_base[cost_key] - cost_a_base[cost_key]) / abs(cost_a_base[cost_key]),
    }


def _cost_diff(factory, region, base_gammas, policy_a_vec, policy_b_vec, delta, cost_key, start_date):
    """cost(A) - cost(B) at a given delta. Positive means A is worse."""
    perturbed = perturb_gammas(base_gammas, delta)
    cost_a = simulate_policy_cost(factory, region, policy_a_vec, perturbed, start_date)
    cost_b = simulate_policy_cost(factory, region, policy_b_vec, perturbed, start_date)
    return cost_a[cost_key] - cost_b[cost_key]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Breakdown analysis for confounding")
    parser.add_argument("--regions", nargs="+", default=["DE", "BR", "ES", "US-NY"])
    parser.add_argument("--startdate", default="2020-03-15")
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--output-dir", default="simulation_results/breakdown_analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  BREAKDOWN ANALYSIS: HOW MUCH CONFOUNDING TO REVERSE CONCLUSIONS?")
    print("=" * 70)

    factory = Pandemic_Factory()
    all_breakdown_results = {}

    for region in args.regions:
        print(f"\n{'='*50}")
        print(f"  Region: {region}")
        print(f"{'='*50}")

        base_gammas, _, _ = get_region_gammas_v2(region)
        print(f"  Base gammas: { {k: round(v, 3) for k, v in sorted(base_gammas.items())} }")

        print("\n  Finding optimal policy at delta=1.0 ...")
        info = find_optimal_and_key_policies(
            factory, region, base_gammas, args.startdate, args.length
        )
        if info is None:
            print("  ERROR: Could not identify optimal policy")
            continue

        optimal_label = info["optimal"]["policy_label"]
        print(f"  Optimal policy: {optimal_label}")
        print(f"  Optimal total cost: {info['optimal']['total']:.2e}")

        region_results = {
            "region": region,
            "optimal_policy": optimal_label,
            "optimal_total_cost": info["optimal"]["total"],
            "comparisons": [],
        }

        for comp_name, label_a, label_b in info["key_comparisons"]:
            print(f"\n  --- {comp_name}: {label_a} vs {label_b} ---")

            vec_a = label_a.split("-")
            vec_b = label_b.split("-")

            result = find_breakdown_delta(
                factory, region, base_gammas, vec_a, vec_b,
                start_date=args.startdate,
            )

            delta_star = result["delta_star"]
            plausible_upper = LITERATURE_CALIBRATION["upper_bound_delta"]
            plausible_central = LITERATURE_CALIBRATION["central_estimate_delta"]

            if result["status"] == "found":
                robust = delta_star > plausible_upper
                result["robust_to_plausible_confounding"] = robust
                result["interpretation"] = (
                    f"Confounding must inflate NPI effects by >{100*(delta_star-1):.0f}% "
                    f"to reverse this comparison. "
                    f"{'This exceeds' if robust else 'This falls within'} the plausible "
                    f"range ({100*(plausible_central-1):.0f}% central, "
                    f"{100*(plausible_upper-1):.0f}% upper bound)."
                )
                print(f"    delta* = {delta_star:.3f} "
                      f"(confounding must inflate NPI effect by >{100*(delta_star-1):.0f}%)")
                print(f"    {'ROBUST' if robust else 'FRAGILE'} to plausible confounding")
            elif result["status"] == "already_reversed":
                print(f"    A is already worse at delta=1.0; no reversal needed")
                result["interpretation"] = "The cost ordering already favors B at baseline."
            else:
                print(f"    Reversal not found in tested range [0.5, 3.0]; very robust")
                result["interpretation"] = (
                    "The comparison is extremely robust: even very large confounding "
                    "does not reverse the cost ordering."
                )
                result["robust_to_plausible_confounding"] = True

            result["comparison_name"] = comp_name
            result["policy_a"] = label_a
            result["policy_b"] = label_b
            region_results["comparisons"].append(result)

        all_breakdown_results[region] = region_results

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    summary_rows = []
    for region, rr in all_breakdown_results.items():
        for comp in rr["comparisons"]:
            summary_rows.append({
                "region": region,
                "comparison": comp["comparison_name"],
                "policy_a": comp["policy_a"],
                "policy_b": comp["policy_b"],
                "delta_star": comp.get("delta_star", float("inf")),
                "margin_pct": comp.get("margin_pct", None),
                "robust": comp.get("robust_to_plausible_confounding", None),
                "status": comp["status"],
            })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(output_dir / "breakdown_summary.csv", index=False)

    # --- Literature calibration ---
    print("\n  LITERATURE CALIBRATION:")
    print(f"    Plausible delta range: [{LITERATURE_CALIBRATION['lower_bound_delta']}, "
          f"{LITERATURE_CALIBRATION['upper_bound_delta']}]")
    print(f"    Central estimate: {LITERATURE_CALIBRATION['central_estimate_delta']}")
    print(f"    Interpretation: {LITERATURE_CALIBRATION['interpretation']}")

    # --- Plot ---
    if len(summary_df) > 0:
        _plot_breakdown_summary(summary_df, output_dir)

    # --- Save full results ---
    full_output = {
        "literature_calibration": LITERATURE_CALIBRATION,
        "regions": all_breakdown_results,
    }
    with open(output_dir / "full_results.json", "w") as f:
        json.dump(full_output, f, indent=2, default=str)

    print(f"\n  All outputs saved to: {output_dir}")
    print("=" * 70)


def _plot_breakdown_summary(summary_df, output_dir):
    """Bar chart of breakdown deltas with plausible range overlay."""
    plot_df = summary_df[summary_df["status"] == "found"].copy()
    if len(plot_df) == 0:
        return

    plot_df["label"] = plot_df["region"] + "\n" + plot_df["comparison"]

    fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 1.5), 6))

    x = np.arange(len(plot_df))
    bars = ax.bar(x, plot_df["delta_star"] - 1.0, bottom=1.0,
                  color=["#2c7fb8" if r else "#d95f02" for r in plot_df["robust"]],
                  edgecolor="white", width=0.6)

    ax.axhline(y=LITERATURE_CALIBRATION["upper_bound_delta"], color="red",
               linestyle="--", alpha=0.7, label=f"Plausible upper ({LITERATURE_CALIBRATION['upper_bound_delta']})")
    ax.axhline(y=LITERATURE_CALIBRATION["central_estimate_delta"], color="orange",
               linestyle="-.", alpha=0.7, label=f"Plausible central ({LITERATURE_CALIBRATION['central_estimate_delta']})")
    ax.axhspan(LITERATURE_CALIBRATION["lower_bound_delta"],
               LITERATURE_CALIBRATION["upper_bound_delta"],
               alpha=0.1, color="red", label="Plausible range")
    ax.axhline(y=1.0, color="gray", linestyle="-", alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Breakdown $\\delta^*$", fontsize=12)
    ax.set_title("Confounding Required to Reverse Key Policy Conclusions", fontsize=13)
    ax.legend(fontsize=9)

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(i, row["delta_star"] + 0.02, f"{row['delta_star']:.2f}",
                ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "breakdown_summary.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "breakdown_summary.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
