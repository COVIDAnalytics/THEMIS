"""
Re-run the scatter-plot analysis using rank-1 ALS gamma estimates.

Computes a FRESH rank-1 gamma matrix via non-negative ALS (the same
procedure as in analyze_gamma_rank.py) and injects those gammas into
the Pandemic_Factory cache so the DELPHI simulations use the new
estimates without any core-code changes.

Key design choices vs the original pipeline:
  - Observed gamma entries are PRESERVED; only unobserved region-policy
    pairs are filled via the rank-1 model (gamma_{R,i} = k_R * g_i).
  - sample_gammas is DISABLED because the Pandemic class's internal
    sampling calls get_region_gammas_v2 directly (bypassing the cache),
    which would mix old-method samples with rank-1 point estimates.
    Confidence intervals still come from DELPHI's own error model.

Produces simulation CSVs and the updated scatter_plot_rank1.pdf.
"""

import os
import sys
import itertools
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd

from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.pandemic import Pandemic_Factory
from pandemic_functions.pandemic_params import region_symbol_country_dict
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    get_region_gammas_v2,
)
from policy_functions.policy import Policy
from utils.visualization_utils import (
    region_policy_scatter_plot_panel,
    shorten_policy_string,
)
from analyze_gamma_rank import build_gamma_matrix, rank1_imputation, POLICY_NAMES

# ── Configuration ─────────────────────────────────────────────────────────
REGIONS = ["BR", "DE", "US-NY", "ES"]
START_DATE = "2020-03-15"
END_DATE = "2020-06-15"
POLICY_LENGTH = 3
# sample_gammas is OFF: the Pandemic class's internal sampling bypasses
# our cache and falls back to the old get_region_gammas_v2 distribution,
# creating a mismatch.  DELPHI's own CIs are still included.
SAMPLE_GAMMAS = False

OUTPUT_DIR = "simulation_results/rank1_scatter"

FUTURE_POLICIES = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Mass_Gatherings_Authorized_But_Others_Restricted",
    "Restrict_Mass_Gatherings_and_Schools",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]


# ── Helpers ───────────────────────────────────────────────────────────────

def _region_to_matrix_key(region: str) -> str:
    """Map a region code (e.g. 'BR') to the gamma-matrix row key."""
    country, province = region_symbol_country_dict[region]
    return f"{country}__{province}".replace(" ", "_")


def compute_fresh_rank1_gammas() -> dict:
    """
    Build the observed gamma matrix from scratch, run rank-1 ALS,
    and return per-region gamma dicts with observed entries preserved.
    All regions use the rank-1 estimates consistently.
    """
    print(f"  Building observed gamma matrix for {START_DATE} to {END_DATE} ...")
    gamma_matrix, obs_mask, region_ids, policy_names = build_gamma_matrix(
        start_date=START_DATE, end_date=END_DATE,
    )
    n_obs = obs_mask.sum()
    n_total = gamma_matrix.size
    print(f"  Matrix: {gamma_matrix.shape[0]} regions x {gamma_matrix.shape[1]} policies, "
          f"{n_obs}/{n_total} observed ({100*n_obs/n_total:.1f}%)")

    print("  Running rank-1 ALS ...")
    completed, k_R = rank1_imputation(gamma_matrix, obs_mask)

    obs_preserved = np.allclose(completed[obs_mask], gamma_matrix[obs_mask], atol=1e-8)
    print(f"  Observed entries preserved: {obs_preserved}")

    rank1 = {}
    for region in REGIONS:
        key = _region_to_matrix_key(region)
        if key not in region_ids:
            raise KeyError(f"Region '{key}' not in gamma matrix ({len(region_ids)} regions)")
        idx = region_ids.index(key)
        gammas = {p: completed[idx, j] for j, p in enumerate(policy_names)}
        obs_flags = {p: bool(obs_mask[idx, j]) for j, p in enumerate(policy_names)}
        rank1[region] = gammas
        print(f"  {region} (k_R={k_R[idx]:.4f}):")
        for p in sorted(gammas):
            tag = " *" if obs_flags[p] else "  "
            print(f"    {tag}{p:55s} {gammas[p]:.6f}")

    return rank1


def preload_factory(factory: Pandemic_Factory, rank1_gammas: dict) -> None:
    """
    Pre-populate the factory's caches with totalcases data and rank-1 gammas
    so that compute_delphi() never falls back to get_region_gammas_v2().
    """
    for region in REGIONS:
        country, province = region_symbol_country_dict[region]
        country_sub = country.replace(" ", "_")
        province_sub = province.replace(" ", "_")

        csv_path = f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing cases file: {csv_path}")

        totalcases = pd.read_csv(csv_path)
        # The factory's cache check uses country_sub but reads with region;
        # populate under both keys to ensure cache hits.
        factory.d_read_data_total_cases[country_sub] = totalcases
        factory.d_read_data_total_cases[region] = totalcases
        factory.d_region_policy_gammas[country_sub] = rank1_gammas[region]
        factory.d_region_policy_gammas[region] = rank1_gammas[region]


# ── Simulation functions (mirroring main.py) ──────────────────────────────

def simulate_actual(factory, region):
    policy = Policy(policy_type="actual", start_date=START_DATE, policy_length=POLICY_LENGTH)
    pandemic = factory.compute_delphi(policy, region=region)
    cost = PandemicCost(pandemic)
    return "actual", cost.__dict__


def simulate_policy(args):
    factory, region, policy_vect = args
    policy = Policy(policy_type="hypothetical", start_date=START_DATE, policy_vector=policy_vect)
    pandemic = factory.compute_delphi(policy, region=region, sample_gammas=SAMPLE_GAMMAS)
    cost = PandemicCost(pandemic)
    return "-".join(policy_vect), cost.__dict__


def run_region(factory, region) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  Region: {region}")
    print(f"{'='*60}")

    rows = []

    print("  Running actual policy ...")
    rows.append(simulate_actual(factory, region))

    scenarios = [(factory, region, list(t))
                 for t in itertools.product(FUTURE_POLICIES, repeat=POLICY_LENGTH)]
    total = len(scenarios)
    print(f"  Running {total} hypothetical scenarios ...")

    n_proc = max(1, os.cpu_count() // 2)
    if n_proc > 1:
        with mp.Pool(processes=n_proc) as pool:
            results = pool.map(simulate_policy, scenarios)
        rows.extend(results)
    else:
        for i, scenario in enumerate(scenarios, 1):
            try:
                rows.append(simulate_policy(scenario))
                if i % 50 == 0:
                    print(f"    {i}/{total} done")
            except Exception:
                print(f"    Error on scenario {scenario[2]}: {sys.exc_info()[1]}")

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


# ── Scatter plot generation ───────────────────────────────────────────────

def build_plots(combined: pd.DataFrame, output_dir: str) -> None:
    from utils.visualization_utils import best_policy_cost_breakdown_plot_panel

    combined["start_date"] = pd.to_datetime(combined["start_date"])
    combined["life_costs"] = combined.d_costs + combined.h_costs + combined.mh_costs
    combined["life_costs_lb"] = combined.d_costs_lb + combined.h_costs_lb + combined.mh_costs_lb
    combined["life_costs_ub"] = combined.d_costs_ub + combined.h_costs_ub + combined.mh_costs_ub
    combined["total_cost"] = combined.life_costs + combined.st_economic_costs
    combined["short_policy_name"] = [
        pname if pname == "actual" else shorten_policy_string(pname)
        for pname in combined["policy"]
    ]

    os.makedirs(output_dir, exist_ok=True)

    fig = region_policy_scatter_plot_panel(
        combined, start_date="2020-03-15", y_val="life_costs"
    )
    fig.write_image(os.path.join(output_dir, "scatter_plot.pdf"))
    print(f"  Saved {output_dir}/scatter_plot.pdf")

    fig2 = best_policy_cost_breakdown_plot_panel(
        combined, n=20, start_date="2020-03-15"
    )
    fig2.write_image(os.path.join(output_dir, "best_policies.pdf"))
    print(f"  Saved {output_dir}/best_policies.pdf")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Computing fresh rank-1 gammas (ALS) ...")
    rank1_gammas = compute_fresh_rank1_gammas()

    print("\nInitializing Pandemic_Factory ...")
    factory = Pandemic_Factory()
    preload_factory(factory, rank1_gammas)

    region_dfs = []
    for region in REGIONS:
        df = run_region(factory, region)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(OUTPUT_DIR, f"test_result_{region}_rank1_{timestamp}.csv")
        df.to_csv(csv_path)
        print(f"  Saved {csv_path}  ({len(df)} rows)")
        region_dfs.append(df)

    combined = pd.concat(region_dfs, axis=0, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "combined_rank1_results.csv"), index=False)

    print("\nGenerating plots ...")
    build_plots(combined, OUTPUT_DIR)
    build_plots(combined, "simulation_results")

    print("\nDone.")
