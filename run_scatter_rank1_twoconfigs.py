"""
Generate scatter plots using rank-1 ALS gammas under two configurations:
  Config 1: 2020-03-15 to 2020-06-15, min_policy_days=20
  Config 2: 2020-03-01 to 2020-07-31, min_policy_days=20
"""

import os
import sys
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.pandemic import Pandemic_Factory
from pandemic_functions.pandemic_params import region_symbol_country_dict
from policy_functions.policy import Policy
from utils.visualization_utils import (
    region_policy_scatter_plot_panel,
    shorten_policy_string,
)
from analyze_gamma_rank import build_gamma_matrix, rank1_imputation, POLICY_NAMES

REGIONS = ["BR", "DE", "US-NY", "ES"]
SIM_START = "2020-03-15"
POLICY_LENGTH = 3
SAMPLE_GAMMAS = False

FUTURE_POLICIES = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Restrict_Mass_Gatherings_and_Schools",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]

CONFIGS = [
    {
        "label": "config1_narrow20",
        "gamma_start": "2020-03-15",
        "gamma_end": "2020-06-15",
        "min_days": 20,
    },
    {
        "label": "config2_wide20",
        "gamma_start": "2020-03-01",
        "gamma_end": "2020-07-31",
        "min_days": 20,
    },
]


def region_key(region_code):
    country, province = region_symbol_country_dict[region_code]
    return f"{country}__{province}".replace(" ", "_")


def compute_rank1_gammas(cfg):
    print(f"  Building gamma matrix: {cfg['gamma_start']} to {cfg['gamma_end']}, "
          f"min_days={cfg['min_days']}")
    gm, om, rids, pnames = build_gamma_matrix(
        start_date=cfg["gamma_start"],
        end_date=cfg["gamma_end"],
        min_policy_days=cfg["min_days"],
    )
    print(f"  Matrix: {gm.shape[0]}x{gm.shape[1]}, "
          f"{om.sum()}/{gm.size} observed ({100*om.sum()/gm.size:.1f}%)")

    completed, k_R = rank1_imputation(gm, om)

    rank1 = {}
    for rc in REGIONS:
        key = region_key(rc)
        if key not in rids:
            raise KeyError(f"{key} not in matrix")
        idx = rids.index(key)
        rank1[rc] = {p: float(completed[idx, j]) for j, p in enumerate(pnames)}
        obs = {p: bool(om[idx, j]) for j, p in enumerate(pnames)}
        print(f"  {rc} (k_R={k_R[idx]:.4f}):")
        for p in sorted(rank1[rc]):
            tag = "*" if obs[p] else " "
            print(f"    {tag} {p:<55s} {rank1[rc][p]:.6f}")
    return rank1


def preload_factory(factory, gammas):
    for rc in REGIONS:
        country, province = region_symbol_country_dict[rc]
        csub = country.replace(" ", "_")
        psub = province.replace(" ", "_")
        csv_path = f"pandemic_functions/pandemic_data/Cases_{csub}_{psub}.csv"
        tc = pd.read_csv(csv_path)
        factory.d_read_data_total_cases[csub] = tc
        factory.d_read_data_total_cases[rc] = tc
        factory.d_region_policy_gammas[csub] = gammas[rc]
        factory.d_region_policy_gammas[rc] = gammas[rc]


def simulate_actual(factory, region):
    policy = Policy(policy_type="actual", start_date=SIM_START,
                    policy_length=POLICY_LENGTH)
    pandemic = factory.compute_delphi(policy, region=region)
    return "actual", PandemicCost(pandemic).__dict__


def simulate_policy(factory, region, pvec):
    policy = Policy(policy_type="hypothetical", start_date=SIM_START,
                    policy_vector=pvec)
    pandemic = factory.compute_delphi(policy, region=region,
                                       sample_gammas=SAMPLE_GAMMAS)
    return "-".join(pvec), PandemicCost(pandemic).__dict__


def run_region(factory, region):
    rows = [simulate_actual(factory, region)]
    scenarios = list(itertools.product(FUTURE_POLICIES, repeat=POLICY_LENGTH))
    total = len(scenarios)
    for i, pvec in enumerate(scenarios, 1):
        try:
            rows.append(simulate_policy(factory, region, list(pvec)))
        except Exception as e:
            print(f"    Error on {pvec}: {e}")
        if i % 50 == 0:
            print(f"    {region}: {i}/{total} done")

    df = pd.DataFrame.from_dict(dict(rows), orient="index")
    if "policy" in df.columns:
        del df["policy"]
    df.reset_index(inplace=True)
    df.rename(columns={"index": "policy"}, inplace=True)
    df["country"] = region
    df["start_date"] = SIM_START
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
    return df[existing]


def build_scatter(combined, out_dir, label):
    combined = combined.copy()
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
    fig = region_policy_scatter_plot_panel(
        combined, start_date="3/15/2020", y_val="life_costs"
    )
    pdf = os.path.join(out_dir, f"scatter_plot_{label}.pdf")
    png = os.path.join(out_dir, f"scatter_plot_{label}.png")
    fig.write_image(pdf)
    fig.write_image(png)
    print(f"  Saved {pdf}")


if __name__ == "__main__":
    base_dir = "simulation_results/rank1_twoconfigs"
    os.makedirs(base_dir, exist_ok=True)

    for cfg in CONFIGS:
        label = cfg["label"]
        out_dir = os.path.join(base_dir, label)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  {label}: gamma window {cfg['gamma_start']}--{cfg['gamma_end']}, "
              f"min_days={cfg['min_days']}")
        print(f"{'='*70}")

        gammas = compute_rank1_gammas(cfg)

        factory = Pandemic_Factory()
        preload_factory(factory, gammas)

        region_dfs = []
        for rc in REGIONS:
            print(f"\n  Simulating {rc} ...")
            df = run_region(factory, rc)
            df.to_csv(os.path.join(out_dir, f"test_result_{rc}.csv"))
            region_dfs.append(df)
            print(f"    {rc}: {len(df)} rows")

        combined = pd.concat(region_dfs, ignore_index=True)
        combined.to_csv(os.path.join(out_dir, "combined.csv"), index=False)

        print(f"\n  Generating scatter plot for {label} ...")
        build_scatter(combined, out_dir, label)

    print("\nAll done.")
