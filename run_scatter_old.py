"""
Reproduce scatter_plot.pdf using the ORIGINAL gamma estimation method
(get_region_gammas_v2 with default dates 2020-03-01 to 2020-07-31,
 policy_days_thresh=20).

This is the exact method used by main.py / Pandemic_Factory to produce
the CSVs that feed the original scatter_plot.pdf in the notebook.
"""

import os
import sys
import json
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.pandemic import Pandemic_Factory
from pandemic_functions.pandemic_params import (
    region_symbol_country_dict,
    future_policies,
    default_dict_normalized_policy_gamma,
)
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    get_region_gammas_v2,
)
from policy_functions.policy import Policy
from utils.visualization_utils import (
    region_policy_scatter_plot_panel,
    shorten_policy_string,
)

REGIONS = ["BR", "DE", "US-NY", "ES"]
START_DATE = "2020-03-15"
POLICY_LENGTH = 3
SAMPLE_GAMMAS = False
OUTPUT_DIR = "simulation_results/old_scatter"

FUTURE_POLICIES = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Restrict_Mass_Gatherings_and_Schools",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]


def simulate_actual(factory, region):
    policy = Policy(
        policy_type="actual", start_date=START_DATE, policy_length=POLICY_LENGTH
    )
    pandemic = factory.compute_delphi(policy, region=region)
    cost = PandemicCost(pandemic)
    return "actual", cost.__dict__


def simulate_policy(factory, region, policy_vect):
    policy = Policy(
        policy_type="hypothetical",
        start_date=START_DATE,
        policy_vector=policy_vect,
    )
    pandemic = factory.compute_delphi(
        policy, region=region, sample_gammas=SAMPLE_GAMMAS
    )
    cost = PandemicCost(pandemic)
    return "-".join(policy_vect), cost.__dict__


def run_region(factory, region):
    print(f"\n{'='*60}")
    print(f"  Region: {region}")
    print(f"{'='*60}")

    rows = []
    print("  Running actual policy ...")
    rows.append(simulate_actual(factory, region))

    scenarios = list(itertools.product(FUTURE_POLICIES, repeat=POLICY_LENGTH))
    total = len(scenarios)
    print(f"  Running {total} hypothetical scenarios (sequential) ...")

    for i, pvec in enumerate(scenarios, 1):
        try:
            rows.append(simulate_policy(factory, region, list(pvec)))
            if i % 50 == 0:
                print(f"    {i}/{total} done")
        except Exception as e:
            print(f"    Error on {pvec}: {e}")

    df = pd.DataFrame.from_dict(dict(rows), orient="index")
    if "policy" in df.columns:
        del df["policy"]
    df.reset_index(inplace=True)
    df.rename(columns={"index": "policy"}, inplace=True)
    df["country"] = region
    df["start_date"] = START_DATE
    df["policy_length"] = POLICY_LENGTH

    col_order = [
        "country", "start_date", "policy_length", "policy",
        "st_economic_costs", "st_economic_costs_lb", "st_economic_costs_ub",
        "lt_economic_costs",
        "d_costs", "d_costs_lb", "d_costs_ub",
        "h_costs", "h_costs_lb", "h_costs_ub",
        "mh_costs", "mh_costs_lb", "mh_costs_ub",
        "num_cases", "num_cases_lb", "num_cases_ub",
        "num_deaths", "num_deaths_lb", "num_deaths_ub",
        "hospitalization_days", "hospitalization_days_lb", "hospitalization_days_ub",
        "icu_days", "icu_days_lb", "icu_days_ub",
        "ventilated_days", "ventilated_days_lb", "ventilated_days_ub",
    ]
    existing = [c for c in col_order if c in df.columns]
    df = df[existing]
    return df


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Record gammas from get_region_gammas_v2 ──────────────
    print("=" * 60)
    print("  GAMMA ESTIMATION: get_region_gammas_v2 (original method)")
    print("  Policy data window: 2020-03-01 to 2020-07-31")
    print("  policy_days_thresh: 20")
    print("=" * 60)

    gamma_records = {}
    for region in REGIONS:
        gammas, err, obs_policies = get_region_gammas_v2(region)
        gamma_records[region] = {
            "gammas": {p: float(v) for p, v in gammas.items()},
            "observed_policies": obs_policies if obs_policies else [],
        }
        print(f"\n  {region} gammas:")
        for p in sorted(gammas):
            is_obs = obs_policies and p in obs_policies
            tag = "OBS" if is_obs else "IMP"
            print(f"    [{tag}] {p:55s} {gammas[p]:.6f}")

    gamma_path = os.path.join(OUTPUT_DIR, "old_gamma_records.json")
    with open(gamma_path, "w") as f:
        json.dump(gamma_records, f, indent=2)
    print(f"\n  Gamma records saved to {gamma_path}")

    # ── Step 2: Run simulations ──────────────────────────────────────
    factory = Pandemic_Factory()
    region_dfs = []
    for region in REGIONS:
        df = run_region(factory, region)
        csv_path = os.path.join(OUTPUT_DIR, f"test_result_{region}_old.csv")
        df.to_csv(csv_path)
        print(f"  Saved {csv_path}  ({len(df)} rows)")
        region_dfs.append(df)

    # ── Step 3: Generate scatter plot ────────────────────────────────
    combined = pd.concat(region_dfs, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "combined_old_results.csv"), index=False)

    combined["start_date"] = pd.to_datetime(combined["start_date"])
    combined["life_costs"] = combined.d_costs + combined.h_costs + combined.mh_costs
    combined["short_policy_name"] = [
        pname if pname == "actual" else shorten_policy_string(pname)
        for pname in combined["policy"]
    ]
    combined["is_actual"] = [
        "Actual" if pn == "actual" else "hypothetical"
        for pn in combined["short_policy_name"]
    ]

    print("\n  Generating scatter plot ...")
    fig = region_policy_scatter_plot_panel(
        combined, start_date="3/15/2020", y_val="life_costs"
    )
    fig.write_image(os.path.join(OUTPUT_DIR, "scatter_plot_old.pdf"))
    fig.write_image(os.path.join(OUTPUT_DIR, "scatter_plot_old.png"))
    print(f"  Scatter plot saved to {OUTPUT_DIR}/scatter_plot_old.pdf")

    # ── Step 4: Print gamma comparison table ─────────────────────────
    print("\n" + "=" * 60)
    print("  GAMMA MATRIX FOR 4 REGIONS (old method)")
    print("=" * 60)
    header = f"{'Region':<10} {'Policy':<55} {'Gamma':>10} {'Status':>6}"
    print(header)
    print("-" * len(header))
    for region in REGIONS:
        rec = gamma_records[region]
        for p in sorted(rec["gammas"]):
            is_obs = p in rec["observed_policies"]
            tag = "OBS" if is_obs else "IMP"
            print(f"{region:<10} {p:<55} {rec['gammas'][p]:>10.6f} {tag:>6}")
        print()

    print("\nDone.")
