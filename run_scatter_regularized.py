"""
Rank-1 ALS with ridge regularization on g_i toward calibrated defaults.

Objective: sum_obs (gamma - k_R * g_i)^2  +  lam * sum_i (g_i - default_i)^2

ALS updates:
  k_R:  unchanged  (penalty only on g)
  g_j:  (sum_{obs_j} k_r * gamma_{r,j} + lam * default_j) / (sum_{obs_j} k_r^2 + lam)

Generates scatter plots for multiple lambda values.
"""

import os
import sys
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.pandemic import Pandemic_Factory
from pandemic_functions.pandemic_params import (
    region_symbol_country_dict,
    default_dict_normalized_policy_gamma,
)
from policy_functions.policy import Policy
from utils.visualization_utils import (
    region_policy_scatter_plot_panel,
    shorten_policy_string,
)
from analyze_gamma_rank import build_gamma_matrix, POLICY_NAMES

REGIONS = ["BR", "DE", "US-NY", "ES"]
SIM_START = "2020-03-15"
GAMMA_START = "2020-03-15"
GAMMA_END = "2020-06-15"
MIN_DAYS = 20
POLICY_LENGTH = 3
SAMPLE_GAMMAS = False
OUTPUT_DIR = "simulation_results/rank1_regularized"

FUTURE_POLICIES = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Restrict_Mass_Gatherings_and_Schools",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]

SHORT = {
    "No_Measure": "No Measure",
    "Restrict_Mass_Gatherings": "Restrict Gather",
    "Mass_Gatherings_Authorized_But_Others_Restricted": "Gather Auth Oth Restr",
    "Restrict_Mass_Gatherings_and_Schools": "Restrict Gather+Sch",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others": "Auth Sch Restr Oth",
    "Restrict_Mass_Gatherings_and_Schools_and_Others": "Restrict All",
    "Lockdown": "Lockdown",
}

LAMBDAS = [0, 1, 5, 10, 25, 50, 100]


def regularized_rank1(gamma_matrix, obs_mask, lam, max_iter=500, tol=1e-9):
    """Rank-1 ALS with ridge penalty pulling g_i toward defaults."""
    n_regions, n_policies = gamma_matrix.shape
    default_g = np.array([default_dict_normalized_policy_gamma[p] for p in POLICY_NAMES])

    g = default_g.copy()
    k = np.ones(n_regions)

    for _ in range(max_iter):
        old_k, old_g = k.copy(), g.copy()

        for r in range(n_regions):
            obs_r = np.where(obs_mask[r])[0]
            if len(obs_r) == 0:
                k[r] = 1.0
                continue
            num = gamma_matrix[r, obs_r] @ g[obs_r]
            den = g[obs_r] @ g[obs_r]
            k[r] = max(num / den, 0.0) if den > 0 else 1.0

        for j in range(n_policies):
            obs_j = np.where(obs_mask[:, j])[0]
            if len(obs_j) == 0:
                continue
            num = gamma_matrix[obs_j, j] @ k[obs_j] + lam * default_g[j]
            den = k[obs_j] @ k[obs_j] + lam
            g[j] = max(num / den, 0.0) if den > 0 else g[j]

        change = np.sqrt(np.sum((k - old_k) ** 2) + np.sum((g - old_g) ** 2))
        if change < tol:
            break

    comp = np.outer(k, g)
    np.clip(comp, 0, None, out=comp)
    comp[obs_mask] = gamma_matrix[obs_mask]
    return comp, k, g


def region_key(rc):
    country, province = region_symbol_country_dict[rc]
    return f"{country}__{province}".replace(" ", "_")


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
    for i, pvec in enumerate(scenarios, 1):
        try:
            rows.append(simulate_policy(factory, region, list(pvec)))
        except Exception as e:
            print(f"      Error: {e}")
        if i % 50 == 0:
            print(f"      {region}: {i}/{len(scenarios)}")

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


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building gamma matrix ...")
    gm, om, rids, pnames = build_gamma_matrix(
        start_date=GAMMA_START, end_date=GAMMA_END, min_policy_days=MIN_DAYS
    )
    print(f"  {gm.shape[0]}x{gm.shape[1]}, {om.sum()}/{gm.size} observed "
          f"({100*om.sum()/gm.size:.1f}%)\n")

    default_g = np.array([default_dict_normalized_policy_gamma[p] for p in POLICY_NAMES])

    # ── Phase 1: show g_i and DE gammas for each lambda ──────────
    print("=" * 90)
    print("  g_i (global policy vector) for each lambda")
    print("=" * 90)
    header = f"  {'Policy':<25}"
    for lam in LAMBDAS:
        header += f"  lam={lam:<5}"
    header += "  default"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_results = {}
    for lam in LAMBDAS:
        comp, k, g = regularized_rank1(gm, om, lam)
        all_results[lam] = (comp, k, g)

    for j, p in enumerate(pnames):
        row = f"  {SHORT[p]:<25}"
        for lam in LAMBDAS:
            row += f"  {all_results[lam][2][j]:>8.4f}"
        row += f"  {default_g[j]:>8.4f}"
        print(row)

    print(f"\n{'='*90}")
    print(f"  Germany gammas for each lambda")
    print(f"{'='*90}")
    de_idx = rids.index(region_key("DE"))
    header2 = f"  {'Policy':<25}"
    for lam in LAMBDAS:
        header2 += f"  lam={lam:<5}"
    header2 += "    old_v2"
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    old_de = {
        "Auth Sch Restr Oth": 0.1315, "Lockdown": 0.0422,
        "Gather Auth Oth Restr": 0.1180, "No Measure": 0.1766,
        "Restrict Gather": 0.1542, "Restrict Gather+Sch": 0.0846,
        "Restrict All": 0.0804,
    }
    for j, p in enumerate(pnames):
        s = SHORT[p]
        obs_tag = "*" if om[de_idx, j] else " "
        row = f" {obs_tag}{s:<25}"
        for lam in LAMBDAS:
            comp = all_results[lam][0]
            row += f"  {comp[de_idx, j]:>8.4f}"
        row += f"  {old_de.get(s, 0):>8.4f}"
        print(row)
    print("  (* = observed)\n")

    # ── Phase 2: run scatter plots for each lambda ───────────────
    for lam in LAMBDAS:
        lam_dir = os.path.join(OUTPUT_DIR, f"lam_{lam}")
        os.makedirs(lam_dir, exist_ok=True)

        comp, k, g = all_results[lam]
        gammas = {}
        for rc in REGIONS:
            key = region_key(rc)
            idx = rids.index(key)
            gammas[rc] = {p: float(comp[idx, j]) for j, p in enumerate(pnames)}

        factory = Pandemic_Factory()
        for rc in REGIONS:
            country, province = region_symbol_country_dict[rc]
            csub = country.replace(" ", "_")
            psub = province.replace(" ", "_")
            tc = pd.read_csv(
                f"pandemic_functions/pandemic_data/Cases_{csub}_{psub}.csv"
            )
            factory.d_read_data_total_cases[csub] = tc
            factory.d_read_data_total_cases[rc] = tc
            factory.d_region_policy_gammas[csub] = gammas[rc]
            factory.d_region_policy_gammas[rc] = gammas[rc]

        print(f"  Running simulations for lambda={lam} ...")
        region_dfs = []
        for rc in REGIONS:
            df = run_region(factory, rc)
            region_dfs.append(df)
        combined = pd.concat(region_dfs, ignore_index=True)
        combined.to_csv(os.path.join(lam_dir, "combined.csv"), index=False)

        combined_plot = combined.copy()
        combined_plot["start_date"] = pd.to_datetime(combined_plot["start_date"])
        combined_plot["life_costs"] = (
            combined_plot.d_costs + combined_plot.h_costs + combined_plot.mh_costs
        )
        combined_plot["short_policy_name"] = [
            pn if pn == "actual" else shorten_policy_string(pn)
            for pn in combined_plot["policy"]
        ]
        combined_plot["is_actual"] = [
            "Actual" if pn == "actual" else "hypothetical"
            for pn in combined_plot["short_policy_name"]
        ]

        fig = region_policy_scatter_plot_panel(
            combined_plot, start_date="3/15/2020", y_val="life_costs"
        )
        fig.write_image(os.path.join(lam_dir, f"scatter_lam{lam}.pdf"))
        fig.write_image(os.path.join(lam_dir, f"scatter_lam{lam}.png"))
        print(f"    Saved scatter_lam{lam}.pdf")

    print("\nAll done.")
