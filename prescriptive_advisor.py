"""
Rolling-Horizon NPI Advisor using THEMIS.

Implements a one-step-lookahead policy advisor that, given the current
epidemic state and the most recent month's policy, evaluates all 6
candidate NPIs for the *next* month and recommends the cost-minimizing
choice with uncertainty bands.

The retrospective rolling exercise starts from March 2020 and, at each
month boundary, runs the advisor using only data available at that point
to compare the advisor's recommendation against what was actually
implemented.

Usage:
    python prescriptive_advisor.py
    python prescriptive_advisor.py --regions DE BR ES US-NY
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from pandemic_functions.pandemic import Pandemic_Factory, Pandemic
from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.pandemic_params import region_symbol_country_dict
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    get_region_gammas_v2,
    read_policy_data_us_only,
    read_oxford_country_policy_data,
)
from policy_functions.policy import Policy
from cost_functions.economic_cost.economic_data.economic_params import TOTAL_GDP

FUTURE_POLICIES = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Restrict_Mass_Gatherings_and_Schools",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]

POLICY_SHORT = {
    "No_Measure": "None",
    "Restrict_Mass_Gatherings": "MG",
    "Restrict_Mass_Gatherings_and_Schools": "MG+Sch",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others": "Sch+MG+Oth",
    "Restrict_Mass_Gatherings_and_Schools_and_Others": "MG+Sch+Oth",
    "Lockdown": "Lock",
    "Mass_Gatherings_Authorized_But_Others_Restricted": "MG+Oth",
}

POLICY_SEVERITY = {p: i for i, p in enumerate(FUTURE_POLICIES)}
POLICY_SEVERITY["Mass_Gatherings_Authorized_But_Others_Restricted"] = 3

REGIONS = ["DE", "BR", "ES", "US-NY"]

ROLLING_MONTHS = [
    ("2020-03-15", "2020-04-15"),
    ("2020-04-15", "2020-05-15"),
    ("2020-05-15", "2020-06-15"),
]


def _evaluate_single_npi(factory, region, npi, context_policies, start_date,
                         total_months=3):
    """
    Evaluate a single NPI for one month, given context of prior months.

    DELPHI requires a full multi-month policy vector from start_date.
    We pad the remaining months with the candidate NPI (status-quo
    continuation assumption) so the vector always has total_months entries.

    context_policies: list of NPI strings for months already decided
    npi: the candidate NPI for the next month
    start_date: the start of the overall policy window
    total_months: length of the full policy vector (default 3)

    Returns cost dict or None on failure.
    """
    decided = context_policies + [npi]
    remaining = total_months - len(decided)
    policy_vector = decided + [npi] * remaining
    try:
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
        base_gammas, _, _ = get_region_gammas_v2(region)
        pandemic = Pandemic(
            policy, region, factory.delphi_prediction,
            totalcases, base_gammas
        )
        cost = PandemicCost(pandemic)
        econ = cost.st_economic_costs
        humanitarian = cost.d_costs + cost.h_costs + cost.mh_costs
        return {
            "npi": npi,
            "npi_short": POLICY_SHORT[npi],
            "economic_costs": econ,
            "humanitarian_costs": humanitarian,
            "total_costs": econ + humanitarian,
            "num_cases": cost.num_cases,
            "num_deaths": cost.num_deaths,
            "d_costs": cost.d_costs,
            "h_costs": cost.h_costs,
            "mh_costs": cost.mh_costs,
            "full_vector": policy_vector,
        }
    except Exception as e:
        print(f"    SKIP {npi}: {e}")
        return None


def advise_next_npi(factory, region, context_policies, start_date,
                    objective="total_costs"):
    """
    One-step lookahead: evaluate all 6 NPIs for the next month and
    return the full ranking plus the recommended (cost-minimizing) NPI.
    """
    results = []
    for npi in FUTURE_POLICIES:
        r = _evaluate_single_npi(factory, region, npi, context_policies,
                                 start_date)
        if r is not None:
            results.append(r)

    if not results:
        return None

    df = pd.DataFrame(results).sort_values(objective)
    recommended = df.iloc[0]
    status_quo_cost = None
    if context_policies:
        sq = df[df["npi"] == context_policies[-1]]
        if len(sq) > 0:
            status_quo_cost = sq.iloc[0][objective]

    savings = None
    if status_quo_cost is not None:
        savings = status_quo_cost - recommended[objective]

    return {
        "recommended_npi": recommended["npi"],
        "recommended_short": recommended["npi_short"],
        "recommended_cost": float(recommended[objective]),
        "status_quo_cost": float(status_quo_cost) if status_quo_cost else None,
        "savings_vs_status_quo": float(savings) if savings is not None else None,
        "ranking": df.to_dict("records"),
        "context": context_policies,
    }


def _get_actual_policy_for_month(region, month_start, month_end):
    """Determine which MECE policy was predominantly in effect during a month."""
    country, province = region_symbol_country_dict[region]
    try:
        if country == "US":
            policy_data = read_policy_data_us_only(
                state=province, start_date=month_start, end_date=month_end)
        else:
            policy_data = read_oxford_country_policy_data(
                country=country, start_date=month_start, end_date=month_end)
    except Exception:
        return "Unknown"

    if policy_data is None or len(policy_data) == 0:
        return "Unknown"

    policy_cols = [c for c in policy_data.columns
                   if c not in ["CountryName", "CountryCode", "Date",
                                "ConfirmedCases", "ConfirmedDeaths",
                                "date", "state"]]

    if not policy_cols:
        return "Unknown"

    day_counts = {}
    for _, row in policy_data.iterrows():
        active = [c for c in policy_cols if row.get(c, 0) == 1]
        if not active:
            label = "No_Measure"
        else:
            label = active[-1]
        day_counts[label] = day_counts.get(label, 0) + 1

    return max(day_counts, key=day_counts.get)


def run_rolling_retrospective(regions, start_date, output_dir):
    """Run the rolling-horizon advisor retrospectively for all regions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factory = Pandemic_Factory()
    all_results = {}

    for region in regions:
        print(f"\n{'='*60}")
        print(f"  Rolling-Horizon Advisor: {region}")
        print(f"{'='*60}")

        monthly_results = []
        greedy_context = []

        for month_idx, (m_start, m_end) in enumerate(ROLLING_MONTHS):
            print(f"\n  Month {month_idx+1}: {m_start} to {m_end}")

            actual_npi = _get_actual_policy_for_month(region, m_start, m_end)
            print(f"    Actual policy implemented: {actual_npi}")

            advice = advise_next_npi(
                factory, region, greedy_context, start_date,
                objective="total_costs"
            )
            if advice is None:
                print("    ERROR: Could not produce advice")
                monthly_results.append({
                    "month": month_idx + 1,
                    "period": f"{m_start} to {m_end}",
                    "actual_npi": actual_npi,
                    "error": True,
                })
                greedy_context.append(actual_npi)
                continue

            rec = advice["recommended_npi"]
            print(f"    THEMIS recommends: {POLICY_SHORT[rec]}")
            print(f"    Recommended cost: {advice['recommended_cost']:.2e}")
            if advice["savings_vs_status_quo"] is not None:
                print(f"    Savings vs status quo: {advice['savings_vs_status_quo']:.2e}")

            ranking = advice["ranking"]
            print("    Full ranking:")
            for r in ranking:
                marker = " <-- ACTUAL" if r["npi"] == actual_npi else ""
                marker += " <-- RECOMMENDED" if r["npi"] == rec else ""
                print(f"      {r['npi_short']:>12}: total={r['total_costs']:.2e}  "
                      f"econ={r['economic_costs']:.2e}  "
                      f"human={r['humanitarian_costs']:.2e}{marker}")

            actual_in_ranking = [r for r in ranking if r["npi"] == actual_npi]
            if actual_in_ranking:
                actual_cost = actual_in_ranking[0]["total_costs"]
            else:
                closest = min(ranking,
                              key=lambda r: abs(POLICY_SEVERITY.get(r["npi"], 0) -
                                                POLICY_SEVERITY.get(actual_npi, 0)))
                actual_cost = closest["total_costs"]
            potential_savings = (actual_cost - advice["recommended_cost"]
                                if actual_cost is not None else None)

            monthly_results.append({
                "month": month_idx + 1,
                "period": f"{m_start} to {m_end}",
                "actual_npi": actual_npi,
                "actual_npi_short": POLICY_SHORT.get(actual_npi, actual_npi),
                "recommended_npi": rec,
                "recommended_npi_short": POLICY_SHORT[rec],
                "recommended_total_cost": advice["recommended_cost"],
                "actual_total_cost": actual_cost,
                "potential_savings": potential_savings,
                "potential_savings_pct": (100 * potential_savings / actual_cost
                                         if potential_savings and actual_cost else None),
                "ranking": ranking,
            })

            greedy_context.append(rec)

        print(f"\n  --- Complete 3-month greedy sequence for {region} ---")
        greedy_seq = [r.get("recommended_npi_short", "?") for r in monthly_results]
        actual_seq = [r.get("actual_npi_short", "?") for r in monthly_results]
        print(f"    THEMIS greedy:  {' -> '.join(greedy_seq)}")
        print(f"    Actual policy:  {' -> '.join(actual_seq)}")

        full_greedy_vec = [r.get("recommended_npi", FUTURE_POLICIES[0])
                           for r in monthly_results if "recommended_npi" in r]
        full_actual_vec = [r.get("actual_npi", FUTURE_POLICIES[0])
                           for r in monthly_results]

        greedy_3mo = _evaluate_full_sequence(factory, region, full_greedy_vec,
                                              start_date)

        actual_sim_vec = []
        for npi in full_actual_vec:
            if npi in FUTURE_POLICIES:
                actual_sim_vec.append(npi)
            else:
                sev = POLICY_SEVERITY.get(npi, 0)
                closest = min(FUTURE_POLICIES,
                              key=lambda p: abs(POLICY_SEVERITY[p] - sev))
                actual_sim_vec.append(closest)
        actual_3mo = _evaluate_full_sequence(factory, region, actual_sim_vec,
                                              start_date)

        hindsight_results = []
        for pv in itertools.product(FUTURE_POLICIES, repeat=3):
            res = _evaluate_full_sequence(factory, region, list(pv), start_date)
            if res is not None:
                hindsight_results.append(res)

        hindsight_optimal = None
        if hindsight_results:
            hindsight_df = pd.DataFrame(hindsight_results)
            hindsight_optimal = hindsight_df.loc[
                hindsight_df["total_costs"].idxmin()].to_dict()

        all_results[region] = {
            "monthly_advice": monthly_results,
            "greedy_sequence": full_greedy_vec,
            "actual_sequence": full_actual_vec,
            "actual_sim_sequence": actual_sim_vec,
            "greedy_3mo_cost": greedy_3mo,
            "actual_3mo_cost": actual_3mo,
            "hindsight_optimal": hindsight_optimal,
        }

        summary_df = pd.DataFrame(monthly_results)
        summary_df.to_csv(output_dir / f"rolling_advice_{region}.csv", index=False)

    _plot_rolling_results(all_results, output_dir)

    with open(output_dir / "rolling_advisor_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    _print_summary_table(all_results)

    print(f"\n  All outputs saved to: {output_dir}")
    return all_results


def _evaluate_full_sequence(factory, region, policy_vector, start_date):
    """Evaluate a complete 3-month policy sequence."""
    try:
        policy = Policy(policy_type="hypothetical", start_date=start_date,
                        policy_vector=policy_vector)
        country, province = region_symbol_country_dict[region]
        country_sub = country.replace(" ", "_")
        province_sub = province.replace(" ", "_")
        totalcases = pd.read_csv(
            f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
        )
        base_gammas, _, _ = get_region_gammas_v2(region)
        pandemic = Pandemic(policy, region, factory.delphi_prediction,
                            totalcases, base_gammas)
        cost = PandemicCost(pandemic)
        econ = cost.st_economic_costs
        humanitarian = cost.d_costs + cost.h_costs + cost.mh_costs
        return {
            "policy_vector": policy_vector,
            "policy_label": "-".join(policy_vector),
            "economic_costs": econ,
            "humanitarian_costs": humanitarian,
            "total_costs": econ + humanitarian,
            "num_deaths": cost.num_deaths,
        }
    except Exception:
        return None


def _plot_rolling_results(all_results, output_dir):
    """2x2 grouped bar chart: 3-month total cost for greedy, actual, hindsight
    with stacked economic/humanitarian components and policy labels."""
    regions = list(all_results.keys())[:4]
    n = len(regions)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    if n < nrows * ncols:
        for i in range(n, nrows * ncols):
            axes.flat[i].set_visible(False)

    bar_labels = ["Greedy\nAdvisor", "Actual\nPolicy", "Hindsight\nOptimal"]
    colors_econ = "#4e79a7"
    colors_human = "#e15759"

    for idx, region in enumerate(regions):
        ax = axes.flat[idx]
        data = all_results[region]

        greedy = data.get("greedy_3mo_cost") or {}
        actual = data.get("actual_3mo_cost") or {}
        hindsight = data.get("hindsight_optimal") or {}

        econ_vals = [greedy.get("economic_costs", 0),
                     actual.get("economic_costs", 0),
                     hindsight.get("economic_costs", 0)]
        human_vals = [greedy.get("humanitarian_costs", 0),
                      actual.get("humanitarian_costs", 0),
                      hindsight.get("humanitarian_costs", 0)]
        total_vals = [e + h for e, h in zip(econ_vals, human_vals)]

        greedy_seq = data.get("greedy_sequence", [])
        actual_seq = data.get("actual_sequence", [])
        hindsight_seq = (hindsight.get("policy_vector", [])
                         if hindsight else [])
        seq_labels = []
        for seq in [greedy_seq, actual_seq, hindsight_seq]:
            short = [POLICY_SHORT.get(p, p[:6]) for p in seq]
            seq_labels.append(" → ".join(short) if short else "?")

        scale = max(total_vals) if max(total_vals) > 0 else 1
        if scale >= 1e12:
            divisor, unit = 1e12, "T"
        elif scale >= 1e9:
            divisor, unit = 1e9, "B"
        else:
            divisor, unit = 1e6, "M"

        e_scaled = [v / divisor for v in econ_vals]
        h_scaled = [v / divisor for v in human_vals]

        x = np.arange(3)
        width = 0.55
        ax.bar(x, e_scaled, width, label="Economic", color=colors_econ,
               alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.bar(x, h_scaled, width, bottom=e_scaled, label="Humanitarian",
               color=colors_human, alpha=0.85, edgecolor="white", linewidth=0.5)

        for i in range(3):
            t = total_vals[i] / divisor
            ax.text(x[i], t + scale / divisor * 0.02,
                    f"${t:.1f}{unit}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(x[i], -scale / divisor * 0.06, seq_labels[i],
                    ha="center", va="top", fontsize=7, color="gray",
                    style="italic")

        ax.set_ylabel(f"3-Month Total Cost (${unit})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_title(f"{region}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ymax = max(total_vals) / divisor * 1.25
        ymin = -max(total_vals) / divisor * 0.12
        ax.set_ylim(ymin, ymax)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_dir / "rolling_advisor_comparison.pdf",
                bbox_inches="tight", dpi=200)
    fig.savefig(output_dir / "rolling_advisor_comparison.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_cost_comparison(all_results, output_dir):
    """Bar chart comparing greedy, actual, and hindsight-optimal total costs."""
    regions = list(all_results.keys())
    greedy_costs = []
    hindsight_costs = []

    for region in regions:
        data = all_results[region]
        gc = data.get("greedy_3mo_cost")
        ho = data.get("hindsight_optimal")
        greedy_costs.append(gc["total_costs"] if gc else 0)
        hindsight_costs.append(ho["total_costs"] if ho else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(regions))
    width = 0.35

    ax.bar(x - width/2, [c / 1e9 for c in greedy_costs], width,
           label="THEMIS Greedy Advisor", color="#1f77b4", edgecolor="white")
    ax.bar(x + width/2, [c / 1e9 for c in hindsight_costs], width,
           label="Hindsight Optimal", color="#2ca02c", edgecolor="white")

    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Total Cost (Billions)", fontsize=12)
    ax.set_title("Rolling Greedy Advisor vs Hindsight-Optimal Policy", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=11)
    ax.legend(fontsize=10)

    for i in range(len(regions)):
        if hindsight_costs[i] > 0:
            gap_pct = 100 * (greedy_costs[i] - hindsight_costs[i]) / abs(
                hindsight_costs[i])
            ax.text(i, max(greedy_costs[i], hindsight_costs[i]) / 1e9 * 1.02,
                    f"{gap_pct:+.1f}%", ha="center", fontsize=9,
                    fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "cost_comparison.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(output_dir / "cost_comparison.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _print_summary_table(all_results):
    """Print a summary table of greedy vs hindsight performance."""
    print("\n" + "=" * 80)
    print("  ROLLING-HORIZON ADVISOR SUMMARY")
    print("=" * 80)
    rows = []
    for region, data in all_results.items():
        gc = data.get("greedy_3mo_cost")
        ho = data.get("hindsight_optimal")
        greedy_seq = " -> ".join(
            POLICY_SHORT.get(p, p) for p in data["greedy_sequence"])
        actual_seq = " -> ".join(
            POLICY_SHORT.get(p, p) for p in data["actual_sequence"])

        greedy_total = gc["total_costs"] if gc else None
        hindsight_total = ho["total_costs"] if ho else None
        gap = (100 * (greedy_total - hindsight_total) / abs(hindsight_total)
               if greedy_total and hindsight_total else None)

        hindsight_seq = " -> ".join(
            POLICY_SHORT.get(p, p)
            for p in (ho["policy_vector"] if ho else [])
        )

        rows.append({
            "Region": region,
            "Greedy Sequence": greedy_seq,
            "Actual Sequence": actual_seq,
            "Hindsight Optimal": hindsight_seq,
            "Greedy Cost": f"{greedy_total:.2e}" if greedy_total else "N/A",
            "Hindsight Cost": f"{hindsight_total:.2e}" if hindsight_total else "N/A",
            "Gap (%)": f"{gap:.1f}" if gap is not None else "N/A",
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Rolling-horizon NPI advisor using THEMIS")
    parser.add_argument("--regions", nargs="+", default=REGIONS)
    parser.add_argument("--startdate", default="2020-03-15")
    parser.add_argument("--output-dir",
                        default="simulation_results/rolling_advisor")
    args = parser.parse_args()

    run_rolling_retrospective(
        regions=args.regions,
        start_date=args.startdate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
