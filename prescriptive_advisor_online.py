"""
True Online Rolling-Horizon NPI Advisor.

The previous prescriptive_advisor.py used hindsight-contaminated parameters:
DELPHI epidemic dynamics and policy effectiveness gammas were both fitted
on data through July 2020, even when the advisor was supposed to make a
recommendation in March 2020.

This module implements the *true* online test:

    At each decision date t in {March 15, April 15, May 15} 2020, the
    advisor uses (a) DELPHI parameters from the dated snapshot fitted only
    on data up to t, and (b) policy-effectiveness gammas estimated only
    from policy observations in [start_date, t].

Three strategies are compared:
  - online greedy advisor (uses only data available at decision time)
  - actual policy implemented by the government
  - hindsight-optimal 3-month sequence (uses the full V2 fit; this is what
    a perfect ex-post planner would have chosen and represents the upper
    bound on how good any advisor could be)

If the online advisor matches the hindsight-optimal closely without seeing
the future, that is genuine evidence the advisor is operationally useful.
"""
import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandemic_functions.pandemic import Pandemic_Factory, Pandemic
from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.pandemic_params import region_symbol_country_dict
from pandemic_functions.delphi_functions import DELPHI_model_policy_scenarios as dmps
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    get_region_gammas_v2,
    read_policy_data_us_only,
    read_oxford_country_policy_data,
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

SNAPSHOT_BY_DECISION = {
    "2020-03-15": "pandemic_functions/pandemic_data/Parameters_Global_20200413.csv",
    "2020-04-15": "pandemic_functions/pandemic_data/Parameters_Global_20200515.csv",
    "2020-05-15": "pandemic_functions/pandemic_data/Parameters_Global_20200615.csv",
}

HINDSIGHT_PARAMS = (
    "pandemic_functions/pandemic_data/"
    "Parameters_Global_V2_20200703_with_NY_correction.csv"
)


def _load_snapshot_as_v2(snapshot_path):
    """
    Load an early-pandemic DELPHI parameter snapshot and pad it to the V2
    schema by setting jump-related terms to zero (no jump observed yet).

    The V2 schema has columns:
        Continent, Country, Province, Data Start Date, MAPE, Infection Rate,
        Median Day of Action, Rate of Action, Rate of Death, Mortality Rate,
        Rate of Mortality Rate Decay, Internal Parameter 1, Internal Parameter 2,
        Jump Magnitude, Jump Time, Jump Decay
    """
    df = pd.read_csv(snapshot_path, keep_default_na=False)
    v2_cols = [
        "Continent", "Country", "Province", "Data Start Date", "MAPE",
        "Infection Rate", "Median Day of Action", "Rate of Action",
        "Rate of Death", "Mortality Rate", "Rate of Mortality Rate Decay",
        "Internal Parameter 1", "Internal Parameter 2",
        "Jump Magnitude", "Jump Time", "Jump Decay",
    ]
    for col in v2_cols:
        if col not in df.columns:
            if col == "MAPE":
                df[col] = 10.0
            elif col == "Rate of Death":
                df[col] = 0.1
            elif col == "Rate of Mortality Rate Decay":
                df[col] = 0.0
            elif col == "Jump Magnitude":
                df[col] = 0.0
            elif col == "Jump Time":
                df[col] = 200.0
            elif col == "Jump Decay":
                df[col] = 1.0
            else:
                df[col] = 0.0
    return df[v2_cols]


def _evaluate_with_online_gammas(factory, region, policy_vector, start_date,
                                  online_gammas):
    """Run the THEMIS simulation using a specified gamma dict."""
    try:
        policy = Policy(policy_type="hypothetical", start_date=start_date,
                        policy_vector=policy_vector)
        country, province = region_symbol_country_dict[region]
        country_sub = country.replace(" ", "_")
        province_sub = province.replace(" ", "_")
        totalcases = pd.read_csv(
            f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
        )
        pandemic = Pandemic(policy, region, factory.delphi_prediction,
                            totalcases, online_gammas)
        cost = PandemicCost(pandemic)
        econ = cost.st_economic_costs
        humanitarian = cost.d_costs + cost.h_costs + cost.mh_costs
        return {
            "policy_vector": policy_vector,
            "economic_costs": econ,
            "humanitarian_costs": humanitarian,
            "total_costs": econ + humanitarian,
            "num_cases": cost.num_cases,
            "num_deaths": cost.num_deaths,
            "d_costs": cost.d_costs,
            "h_costs": cost.h_costs,
            "mh_costs": cost.mh_costs,
        }
    except Exception as e:
        print(f"    SKIP {policy_vector}: {e}")
        return None


def _get_online_gammas(region, start_date, decision_date):
    """
    Estimate policy-effectiveness gammas using only data available up to
    decision_date.  Temporarily swaps the global past_parameters in the
    DELPHI module to the dated snapshot.

    Returns: dict {policy_name: gamma}.
    """
    snapshot_path = SNAPSHOT_BY_DECISION[decision_date]
    snapshot_df = _load_snapshot_as_v2(snapshot_path)

    original_params = dmps.past_parameters
    try:
        dmps.past_parameters = snapshot_df
        gammas, _, _ = get_region_gammas_v2(
            region, start_date=start_date, end_date=decision_date,
            policy_days_thresh=5,
        )
    finally:
        dmps.past_parameters = original_params

    return gammas


def _get_hindsight_gammas(region, start_date, end_date):
    """Estimate gammas using the full hindsight DELPHI fit (V2_20200703)."""
    gammas, _, _ = get_region_gammas_v2(
        region, start_date=start_date, end_date=end_date,
        policy_days_thresh=20,
    )
    return gammas


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


def _online_advise(factory, region, context_policies, start_date,
                    decision_date, online_gammas):
    """
    One-step lookahead with online gammas: evaluate all 6 NPIs for the next
    month assuming we continue with that NPI for the remaining months
    (status-quo continuation), and pick the cheapest by total cost.
    """
    results = []
    for npi in FUTURE_POLICIES:
        decided = context_policies + [npi]
        remaining = 3 - len(decided)
        policy_vector = decided + [npi] * remaining
        r = _evaluate_with_online_gammas(factory, region, policy_vector,
                                          start_date, online_gammas)
        if r is None:
            continue
        results.append({
            "npi": npi,
            "npi_short": POLICY_SHORT[npi],
            "economic_costs": r["economic_costs"],
            "humanitarian_costs": r["humanitarian_costs"],
            "total_costs": r["total_costs"],
            "num_deaths": r["num_deaths"],
            "full_vector": policy_vector,
        })
    if not results:
        return None
    df = pd.DataFrame(results).sort_values("total_costs")
    rec = df.iloc[0]
    return {
        "recommended_npi": rec["npi"],
        "recommended_short": rec["npi_short"],
        "recommended_cost": float(rec["total_costs"]),
        "ranking": df.to_dict("records"),
    }


def run_online_rolling_retrospective(regions, start_date, output_dir):
    """
    True online test: at each decision date, the advisor uses gamma
    parameters fitted only on data available up to that date.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factory = Pandemic_Factory()
    all_results = {}

    for region in regions:
        print(f"\n{'='*60}")
        print(f"  Online Rolling-Horizon Advisor: {region}")
        print(f"{'='*60}")

        monthly = []
        online_context = []

        for month_idx, (m_start, m_end) in enumerate(ROLLING_MONTHS):
            print(f"\n  Month {month_idx+1}: decision at {m_start}")

            decision_date = m_start
            print(f"    Estimating online gammas using data {start_date} -> "
                  f"{decision_date} ...")
            online_gammas = _get_online_gammas(region, start_date, decision_date)
            if online_gammas is None or len(online_gammas) == 0:
                print(f"    Could not estimate online gammas; falling back")
                online_gammas = _get_hindsight_gammas(region, start_date,
                                                       decision_date)
            print(f"    Online gammas: " + ", ".join(
                f"{POLICY_SHORT.get(p,p)}={g:.3f}" for p, g in online_gammas.items()
            ))

            actual_npi = _get_actual_policy_for_month(region, m_start, m_end)
            print(f"    Actual policy implemented: "
                  f"{POLICY_SHORT.get(actual_npi, actual_npi)}")

            advice = _online_advise(factory, region, online_context,
                                     start_date, decision_date, online_gammas)
            if advice is None:
                print("    Advisor failed")
                online_context.append(actual_npi if actual_npi in FUTURE_POLICIES
                                       else FUTURE_POLICIES[0])
                monthly.append({"month": month_idx+1, "error": True})
                continue

            rec = advice["recommended_npi"]
            print(f"    Online advisor recommends: {POLICY_SHORT[rec]} "
                  f"(cost={advice['recommended_cost']:.2e})")
            print("    Full ranking:")
            for r in advice["ranking"]:
                marker = ""
                if r["npi"] == rec:
                    marker = " <-- ONLINE"
                if r["npi"] == actual_npi:
                    marker += " <-- ACTUAL"
                print(f"      {r['npi_short']:>12}: total={r['total_costs']:.2e}"
                      f"{marker}")

            monthly.append({
                "month": month_idx+1,
                "decision_date": decision_date,
                "actual_npi": actual_npi,
                "actual_npi_short": POLICY_SHORT.get(actual_npi, actual_npi),
                "online_npi": rec,
                "online_npi_short": POLICY_SHORT[rec],
                "online_cost": advice["recommended_cost"],
                "online_gammas": {p: float(g) for p, g in online_gammas.items()},
                "ranking": advice["ranking"],
            })
            online_context.append(rec)

        online_seq = [m.get("online_npi") for m in monthly if "online_npi" in m]
        actual_seq = [m.get("actual_npi") for m in monthly]

        print(f"\n  --- 3-month sequences for {region} ---")
        print(f"    Online advisor: "
              f"{' -> '.join(POLICY_SHORT.get(p,p) for p in online_seq)}")
        print(f"    Actual policy:  "
              f"{' -> '.join(POLICY_SHORT.get(p,p) for p in actual_seq)}")

        # For final 3-month cost evaluation we use the HINDSIGHT (V2) gammas
        # because that is the ground-truth simulator the paper used; this
        # ensures the comparison is between strategies on a single ground
        # truth, not between simulators.
        hindsight_gammas = _get_hindsight_gammas(region, start_date,
                                                  "2020-07-15")
        online_3mo = _evaluate_with_online_gammas(
            factory, region, online_seq, start_date, hindsight_gammas)

        actual_sim = []
        for npi in actual_seq:
            if npi in FUTURE_POLICIES:
                actual_sim.append(npi)
            else:
                sev = POLICY_SEVERITY.get(npi, 0)
                actual_sim.append(min(FUTURE_POLICIES,
                                       key=lambda p: abs(POLICY_SEVERITY[p] - sev)))
        actual_3mo = _evaluate_with_online_gammas(
            factory, region, actual_sim, start_date, hindsight_gammas)

        print("    Computing hindsight optimum over 216 sequences ...")
        hindsight_results = []
        for pv in itertools.product(FUTURE_POLICIES, repeat=3):
            r = _evaluate_with_online_gammas(factory, region, list(pv),
                                              start_date, hindsight_gammas)
            if r is not None:
                hindsight_results.append(r)
        hindsight_optimal = None
        if hindsight_results:
            hdf = pd.DataFrame(hindsight_results)
            hindsight_optimal = hdf.loc[hdf["total_costs"].idxmin()].to_dict()

        all_results[region] = {
            "monthly": monthly,
            "online_sequence": online_seq,
            "actual_sequence": actual_seq,
            "actual_sim_sequence": actual_sim,
            "online_3mo_cost": online_3mo,
            "actual_3mo_cost": actual_3mo,
            "hindsight_optimal": hindsight_optimal,
        }
        pd.DataFrame(monthly).to_csv(
            output_dir / f"online_advice_{region}.csv", index=False)

    _plot_online_results(all_results, output_dir)

    with open(output_dir / "online_advisor_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    _print_online_summary(all_results)
    print(f"\n  All outputs saved to: {output_dir}")
    return all_results


def _plot_online_results(all_results, output_dir):
    """2x2 grouped-bar comparison: online advisor, actual, hindsight optimal."""
    regions = list(all_results.keys())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, region in enumerate(regions):
        ax = axes.flat[idx]
        data = all_results[region]
        online = data.get("online_3mo_cost") or {}
        actual = data.get("actual_3mo_cost") or {}
        hindsight = data.get("hindsight_optimal") or {}

        econ = [online.get("economic_costs", 0),
                actual.get("economic_costs", 0),
                hindsight.get("economic_costs", 0)]
        human = [online.get("humanitarian_costs", 0),
                 actual.get("humanitarian_costs", 0),
                 hindsight.get("humanitarian_costs", 0)]
        total = [e + h for e, h in zip(econ, human)]

        seqs = [data.get("online_sequence", []),
                data.get("actual_sequence", []),
                hindsight.get("policy_vector", []) if hindsight else []]
        seq_labels = [" → ".join(POLICY_SHORT.get(p, p[:6]) for p in s) if s else "?"
                      for s in seqs]

        scale = max(total) if max(total) > 0 else 1
        if scale >= 1e12:
            div, unit = 1e12, "T"
        elif scale >= 1e9:
            div, unit = 1e9, "B"
        else:
            div, unit = 1e6, "M"

        e_s = [v / div for v in econ]
        h_s = [v / div for v in human]
        x = np.arange(3)
        ax.bar(x, e_s, 0.55, label="Economic", color="#4e79a7",
               alpha=0.85, edgecolor="white")
        ax.bar(x, h_s, 0.55, bottom=e_s, label="Humanitarian",
               color="#e15759", alpha=0.85, edgecolor="white")

        for i in range(3):
            t = total[i] / div
            ax.text(x[i], t + scale/div * 0.02, f"${t:.1f}{unit}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(x[i], -scale/div * 0.06, seq_labels[i],
                    ha="center", va="top", fontsize=7, color="gray",
                    style="italic")

        ax.set_ylabel(f"3-Month Total Cost (${unit})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["Online\nAdvisor", "Actual\nPolicy",
                            "Hindsight\nOptimal"], fontsize=9)
        ax.set_title(region, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-max(total)/div * 0.12, max(total)/div * 1.25)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_dir / "online_advisor_comparison.pdf",
                bbox_inches="tight", dpi=200)
    fig.savefig(output_dir / "online_advisor_comparison.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


def _print_online_summary(all_results):
    print("\n" + "=" * 80)
    print("  TRUE ONLINE ROLLING-HORIZON ADVISOR SUMMARY")
    print("=" * 80)
    rows = []
    for region, data in all_results.items():
        on = data.get("online_3mo_cost")
        ac = data.get("actual_3mo_cost")
        ho = data.get("hindsight_optimal")
        on_seq = " -> ".join(POLICY_SHORT.get(p, p)
                             for p in data["online_sequence"])
        ac_seq = " -> ".join(POLICY_SHORT.get(p, p)
                             for p in data["actual_sequence"])
        ho_seq = " -> ".join(POLICY_SHORT.get(p, p)
                             for p in (ho["policy_vector"] if ho else []))
        on_total = on["total_costs"] if on else None
        ac_total = ac["total_costs"] if ac else None
        ho_total = ho["total_costs"] if ho else None
        gap_online = (100 * (on_total - ho_total) / abs(ho_total)
                       if on_total and ho_total else None)
        gap_actual = (100 * (ac_total - ho_total) / abs(ho_total)
                       if ac_total and ho_total else None)
        rows.append({
            "Region": region,
            "Online Advisor": on_seq,
            "Actual Policy": ac_seq,
            "Hindsight Optimal": ho_seq,
            "Online Cost": f"{on_total:.2e}" if on_total else "N/A",
            "Actual Cost": f"{ac_total:.2e}" if ac_total else "N/A",
            "Hindsight Cost": f"{ho_total:.2e}" if ho_total else "N/A",
            "Online Gap (%)": f"{gap_online:.1f}" if gap_online is not None else "N/A",
            "Actual Gap (%)": f"{gap_actual:.1f}" if gap_actual is not None else "N/A",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="True online rolling-horizon NPI advisor")
    parser.add_argument("--regions", nargs="+", default=REGIONS)
    parser.add_argument("--startdate", default="2020-03-15")
    parser.add_argument("--output-dir",
                        default="simulation_results/online_advisor")
    args = parser.parse_args()
    run_online_rolling_retrospective(
        regions=args.regions,
        start_date=args.startdate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
