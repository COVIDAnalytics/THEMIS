"""
NPI Escalation/De-escalation Trigger Protocol derived from THEMIS.

Distills the full THEMIS simulation framework into a compact, reusable
2D lookup table mapping (case growth rate, hospital capacity utilization)
to the recommended NPI action.  This gives decision-makers a rule-driven
protocol that can be applied without rerunning the simulation.

The protocol is derived by:
  1. Running the full THEMIS simulation grid for each region
  2. Identifying the optimal NPI at each epidemic-state bin
  3. Aggregating across regions to produce universal and region-specific
     trigger thresholds

Usage:
    python prescriptive_trigger_protocol.py
    python prescriptive_trigger_protocol.py --regions DE BR ES US-NY
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

from pandemic_functions.pandemic import Pandemic_Factory, Pandemic
from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.pandemic_params import (
    region_symbol_country_dict, p_h, global_populations,
)
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    get_region_gammas_v2,
)
from policy_functions.policy import Policy
from cost_functions.economic_cost.economic_data.economic_params import (
    TOTAL_GDP, TOTAL_LABOR_FORCE,
)

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
}

POLICY_SEVERITY = {p: i for i, p in enumerate(FUTURE_POLICIES)}
SEVERITY_TO_POLICY = {v: k for k, v in POLICY_SEVERITY.items()}

REGIONS = ["DE", "BR", "ES", "US-NY"]

COST_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _get_population(region):
    """Get population for a region."""
    country, province = region_symbol_country_dict[region]
    pop_row = global_populations[
        (global_populations.Country == country) &
        (global_populations.Province == province)
    ]
    if len(pop_row) > 0:
        return pop_row.pop2016.iloc[-1]
    return 1e7


def _simulate_grid(factory, region, start_date, policy_length):
    """Run full policy grid and augment with epidemic state features."""
    base_gammas, _, _ = get_region_gammas_v2(region)
    country, province = region_symbol_country_dict[region]
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    totalcases = pd.read_csv(
        f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
    )
    population = _get_population(region)
    gdp = TOTAL_GDP.get(region, 1e12)

    scenarios = list(itertools.product(FUTURE_POLICIES, repeat=policy_length))
    rows = []
    for pv in scenarios:
        pv_list = list(pv)
        try:
            policy = Policy(policy_type="hypothetical", start_date=start_date,
                            policy_vector=pv_list)
            pandemic = Pandemic(policy, region, factory.delphi_prediction,
                                totalcases, base_gammas)
            cost = PandemicCost(pandemic)
            econ = cost.st_economic_costs
            humanitarian = cost.d_costs + cost.h_costs + cost.mh_costs

            cases_per_capita = cost.num_cases / population if population else 0
            deaths_per_capita = cost.num_deaths / population if population else 0
            hosp_rate = cost.hospitalization_days / population if population else 0

            month1_gamma = base_gammas.get(pv_list[0], 1.0)
            case_growth_proxy = (1 - month1_gamma) * cases_per_capita

            rows.append({
                "region": region,
                "policy_vector": pv_list,
                "month1": pv_list[0],
                "month2": pv_list[1],
                "month3": pv_list[2],
                "month1_severity": POLICY_SEVERITY[pv_list[0]],
                "economic_costs": econ,
                "humanitarian_costs": humanitarian,
                "total_costs": econ + humanitarian,
                "econ_pct_gdp": 100 * econ / gdp if gdp else 0,
                "num_cases": cost.num_cases,
                "num_deaths": cost.num_deaths,
                "cases_per_capita": cases_per_capita,
                "deaths_per_100k": deaths_per_capita * 1e5,
                "hospitalization_days": cost.hospitalization_days,
                "hosp_rate": hosp_rate,
                "icu_days": cost.icu_days,
                "month1_gamma": month1_gamma,
                "case_growth_proxy": case_growth_proxy,
            })
        except Exception as e:
            continue
    return pd.DataFrame(rows), base_gammas


def _compute_cost_weight_sensitivity(df, region):
    """
    For 11 humanitarian-cost weight values w in [0,1], find the optimal
    3-month NPI sequence by minimizing  w * humanitarian + (1-w) * economic.

    Returns a list of dicts with the weight, optimal NPI sequence, and costs.
    """
    rows = []
    for w in COST_WEIGHTS:
        df["weighted_cost"] = w * df["humanitarian_costs"] + \
                              (1 - w) * df["economic_costs"]
        best_idx = df["weighted_cost"].idxmin()
        best = df.loc[best_idx]
        pv = best["policy_vector"]
        rows.append({
            "region": region,
            "w_humanitarian": w,
            "w_economic": 1 - w,
            "optimal_sequence": [POLICY_SHORT.get(p, p) for p in pv],
            "month1": POLICY_SHORT.get(pv[0], pv[0]),
            "month2": POLICY_SHORT.get(pv[1], pv[1]),
            "month3": POLICY_SHORT.get(pv[2], pv[2]),
            "economic_costs": float(best["economic_costs"]),
            "humanitarian_costs": float(best["humanitarian_costs"]),
            "total_costs": float(best["total_costs"]),
            "deaths_per_100k": float(best["deaths_per_100k"]),
            "econ_pct_gdp": float(best["econ_pct_gdp"]),
        })
    return rows


def _compute_pareto_frontier(df):
    """
    Find the Pareto-efficient NPI sequences: those not dominated
    on both economic and humanitarian cost simultaneously.
    Group by month-1 NPI and find the best (min econ, min humanitarian)
    for each.
    """
    grouped = df.groupby("month1").agg({
        "economic_costs": "min",
        "humanitarian_costs": "min",
        "total_costs": "min",
    }).reset_index()
    grouped["month1_short"] = grouped["month1"].map(POLICY_SHORT)

    econ = grouped["economic_costs"].values
    hum = grouped["humanitarian_costs"].values
    n = len(grouped)
    is_dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if econ[j] <= econ[i] and hum[j] <= hum[i] and \
               (econ[j] < econ[i] or hum[j] < hum[i]):
                is_dominated[i] = True
                break
    grouped["is_pareto"] = [not d for d in is_dominated]
    return grouped


def _plot_pareto_combined(all_sim_data, output_dir):
    """2x2 Pareto frontier: each region shows econ vs humanitarian cost
    for each month-1 NPI, with Pareto-efficient NPIs highlighted."""
    regions = list(all_sim_data.keys())[:4]
    n = len(regions)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    if n < nrows * ncols:
        for i in range(n, nrows * ncols):
            axes.flat[i].set_visible(False)

    npi_colors = {
        "None": "#2ca02c", "MG": "#98df8a", "MG+Sch": "#ffbb78",
        "Sch+MG+Oth": "#ff7f0e", "MG+Sch+Oth": "#d62728", "Lock": "#9467bd",
    }

    for idx, region in enumerate(regions):
        ax = axes.flat[idx]
        df = all_sim_data[region]

        pareto = _compute_pareto_frontier(df)
        scale_e = df["economic_costs"].max()
        scale_h = df["humanitarian_costs"].max()
        if scale_e >= 1e12:
            div_e, unit_e = 1e12, "T"
        elif scale_e >= 1e9:
            div_e, unit_e = 1e9, "B"
        else:
            div_e, unit_e = 1e6, "M"
        if scale_h >= 1e12:
            div_h, unit_h = 1e12, "T"
        elif scale_h >= 1e9:
            div_h, unit_h = 1e9, "B"
        else:
            div_h, unit_h = 1e6, "M"

        for npi_full in FUTURE_POLICIES:
            short = POLICY_SHORT[npi_full]
            subset = df[df["month1"] == npi_full]
            ax.scatter(subset["economic_costs"] / div_e,
                       subset["humanitarian_costs"] / div_h,
                       color=npi_colors.get(short, "gray"),
                       alpha=0.25, s=15, label=None)

        for _, row in pareto.iterrows():
            marker = "*" if row["is_pareto"] else "o"
            size = 200 if row["is_pareto"] else 80
            short = row["month1_short"]
            ax.scatter(row["economic_costs"] / div_e,
                       row["humanitarian_costs"] / div_h,
                       color=npi_colors.get(short, "gray"),
                       s=size, marker=marker, edgecolors="black",
                       linewidths=1.5, zorder=5,
                       label=f"{short}{'*' if row['is_pareto'] else ''}")

        ax.set_xlabel(f"Economic Cost (${unit_e})", fontsize=10)
        ax.set_ylabel(f"Humanitarian Cost (${unit_h})", fontsize=10)
        ax.set_title(f"{region}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", title="Month-1 NPI",
                  title_fontsize=8)
        ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_dir / "pareto_frontiers.pdf", bbox_inches="tight",
                dpi=200)
    fig.savefig(output_dir / "pareto_frontiers.png", bbox_inches="tight",
                dpi=200)
    plt.close(fig)


def _plot_trigger_heatmaps(all_sensitivity, output_dir):
    """2x2 heatmap: rows = w values, columns = month, cells = recommended NPI."""
    regions = list(all_sensitivity.keys())[:4]
    W_DISPLAY = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    month_labels = ["Month 1", "Month 2", "Month 3"]

    n = len(regions)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
    if n < nrows * ncols:
        for i in range(n, nrows * ncols):
            axes.flat[i].set_visible(False)

    cmap = plt.cm.RdYlGn_r

    for idx, region in enumerate(regions):
        ax = axes.flat[idx]
        sens = all_sensitivity[region]

        grid = np.zeros((len(W_DISPLAY), 3))
        text_grid = [[""] * 3 for _ in range(len(W_DISPLAY))]

        for wi, w in enumerate(W_DISPLAY):
            row = next((s for s in sens if abs(s["w_humanitarian"] - w) < 0.01),
                       None)
            if row is None:
                continue
            for mi, mkey in enumerate(["month1", "month2", "month3"]):
                npi_short = row[mkey]
                sev = POLICY_SEVERITY.get(
                    next((k for k, v in POLICY_SHORT.items() if v == npi_short),
                         "No_Measure"),
                    0)
                grid[wi, mi] = sev
                text_grid[wi][mi] = npi_short

        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(month_labels, fontsize=10)
        ax.set_yticks(range(len(W_DISPLAY)))
        ax.set_yticklabels([f"w={w:.1f}" for w in W_DISPLAY], fontsize=9)
        ax.set_ylabel("Humanitarian Cost Weight (w)", fontsize=10)
        ax.set_title(f"{region}", fontsize=12, fontweight="bold")

        for wi in range(len(W_DISPLAY)):
            for mi in range(3):
                ax.text(mi, wi, text_grid[wi][mi], ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if grid[wi, mi] >= 3 else "black")

    cbar = fig.colorbar(axes.flat[0].images[0], ax=axes.flat[-1],
                        label="NPI Severity", shrink=0.8)
    cbar.set_ticks(range(6))
    cbar.set_ticklabels([POLICY_SHORT[p] for p in FUTURE_POLICIES])

    plt.tight_layout(pad=2.0)
    fig.savefig(output_dir / "trigger_heatmap_w_month.pdf",
                bbox_inches="tight", dpi=200)
    fig.savefig(output_dir / "trigger_heatmap_w_month.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


def run_trigger_protocol(regions, start_date, policy_length, output_dir):
    """Main entry: Pareto frontier + cost-weight sensitivity analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factory = Pandemic_Factory()
    all_sim_data = {}
    all_sensitivity = {}
    all_pareto = {}

    for region in regions:
        print(f"\n{'='*60}")
        print(f"  Prescriptive Protocol: {region}")
        print(f"{'='*60}")

        print("  Simulating policy grid ...")
        df, base_gammas = _simulate_grid(factory, region, start_date,
                                          policy_length)
        print(f"  {len(df)} simulations completed")
        all_sim_data[region] = df

        print("  Computing cost-weight sensitivity ...")
        sens = _compute_cost_weight_sensitivity(df, region)
        all_sensitivity[region] = sens

        print("\n  COST-WEIGHT SENSITIVITY TABLE:")
        print(f"  {'w_hum':>5} {'w_econ':>6} | {'Month1':>10} {'Month2':>10} "
              f"{'Month3':>10} | {'Econ($B)':>10} {'Human($B)':>10} "
              f"{'Deaths/100k':>12}")
        print("  " + "-" * 90)
        for s in sens:
            seq = s["optimal_sequence"]
            print(f"  {s['w_humanitarian']:5.1f} {s['w_economic']:6.1f} | "
                  f"{seq[0]:>10} {seq[1]:>10} {seq[2]:>10} | "
                  f"{s['economic_costs']/1e9:10.1f} "
                  f"{s['humanitarian_costs']/1e9:10.1f} "
                  f"{s['deaths_per_100k']:12.1f}")

        pareto = _compute_pareto_frontier(df)
        all_pareto[region] = pareto
        print("\n  PARETO-EFFICIENT MONTH-1 NPIs:")
        pareto_eff = pareto[pareto["is_pareto"]]
        for _, row in pareto_eff.iterrows():
            print(f"    {row['month1_short']:>12}: Econ={row['economic_costs']:.2e}, "
                  f"Humanitarian={row['humanitarian_costs']:.2e}")

        unique_m1 = set(s["month1"] for s in sens)
        print(f"\n  Distinct month-1 NPIs across weights: {sorted(unique_m1)}")

        df.to_csv(output_dir / f"simulations_{region}.csv", index=False)

    _plot_pareto_combined(all_sim_data, output_dir)
    _plot_trigger_heatmaps(all_sensitivity, output_dir)

    _print_protocol_summary(all_sensitivity, all_pareto, output_dir)

    full_output = {
        "regions": {
            region: {
                "sensitivity": all_sensitivity[region],
                "pareto": all_pareto[region].to_dict("records"),
            }
            for region in regions
        },
    }
    with open(output_dir / "trigger_protocol_full.json", "w") as f:
        json.dump(full_output, f, indent=2, default=str)

    print(f"\n  All outputs saved to: {output_dir}")
    return all_sensitivity


def _print_protocol_summary(all_sensitivity, all_pareto, output_dir):
    """Print the 2D trigger table and if-then rules."""
    W_DISPLAY = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    print("\n" + "=" * 70)
    print("  2D NPI ESCALATION / DE-ESCALATION PROTOCOL")
    print("=" * 70)
    print("\n  Rows = humanitarian cost weight w")
    print("  Columns = month position in the pandemic")
    print("  Cells = optimal NPI from THEMIS simulations\n")

    for region, sens in all_sensitivity.items():
        print(f"\n  --- {region} ---")
        header = f"  {'w':>5} | {'Month 1':>12} {'Month 2':>12} {'Month 3':>12}"
        print(header)
        print("  " + "-" * 50)
        for w in W_DISPLAY:
            row = next((s for s in sens if abs(s["w_humanitarian"] - w) < 0.01),
                       None)
            if row is None:
                continue
            print(f"  {w:5.1f} | {row['month1']:>12} {row['month2']:>12} "
                  f"{row['month3']:>12}")

    print("\n\n  IF-THEN RULES (cross-regional):")
    print("  " + "-" * 70)
    for w in W_DISPLAY:
        rules_at_w = {}
        for region, sens in all_sensitivity.items():
            row = next((s for s in sens if abs(s["w_humanitarian"] - w) < 0.01),
                       None)
            if row:
                rules_at_w[region] = row["optimal_sequence"]

        if not rules_at_w:
            continue
        seqs = list(rules_at_w.values())
        all_same = all(s == seqs[0] for s in seqs)
        if all_same:
            seq = " -> ".join(seqs[0])
            print(f"    IF w={w:.1f} THEN all regions: {seq}")
        else:
            for region, seq in rules_at_w.items():
                s = " -> ".join(seq)
                print(f"    IF w={w:.1f} AND region={region} THEN {s}")

    print("\n\n  ESCALATION / DE-ESCALATION PATTERNS:")
    print("  " + "-" * 70)
    for region, sens in all_sensitivity.items():
        w5 = next((s for s in sens if abs(s["w_humanitarian"] - 0.5) < 0.01),
                  None)
        if w5:
            seq = w5["optimal_sequence"]
            sev = [POLICY_SEVERITY.get(
                       next((k for k, v in POLICY_SHORT.items() if v == npi),
                            "No_Measure"), 0) for npi in seq]
            if sev[0] > sev[-1]:
                pattern = "DE-ESCALATION"
            elif sev[0] < sev[-1]:
                pattern = "ESCALATION"
            else:
                pattern = "CONSTANT"
            print(f"    {region} at w=0.5: {' -> '.join(seq)} ({pattern})")

    summary_rows = []
    for region, sens in all_sensitivity.items():
        m1_set = set(s["month1"] for s in sens)
        full_set = set()
        for s in sens:
            full_set.update(s["optimal_sequence"])
        summary_rows.append({
            "Region": region,
            "Distinct M1 NPIs": len(m1_set),
            "Total Distinct NPIs": len(full_set),
            "w=0": sens[0]["month1"],
            "w=0.5": next(s for s in sens
                          if abs(s["w_humanitarian"] - 0.5) < 0.01)["month1"],
            "w=1": sens[-1]["month1"],
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\n  SUMMARY:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(output_dir / "protocol_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Derive NPI prescriptive protocol from THEMIS")
    parser.add_argument("--regions", nargs="+", default=REGIONS)
    parser.add_argument("--startdate", default="2020-03-15")
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--output-dir",
                        default="simulation_results/trigger_protocol")
    args = parser.parse_args()

    run_trigger_protocol(
        regions=args.regions,
        start_date=args.startdate,
        policy_length=args.length,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
