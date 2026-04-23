"""
Granular policy partition sensitivity analysis for THEMIS.

Tests whether the 7-category MECE policy partition is robust to a finer
decomposition that preserves individual Oxford OxCGRT sub-indicators instead
of collapsing them into composite groups.

The original 7 categories collapse C3/C4/C5 -> "Restrict_Mass_Gatherings"
and C2/C7/C8 -> "Others".  This script constructs a finer partition that
keeps the individual binary features as separate dimensions, producing
more granular combinatorial categories.  It then:

  1. Builds the granular MECE policy panel for all regions in past_parameters
  2. Computes granular gamma_{R,i} values and the rank-1 model
  3. For 4 paper regions, compares DELPHI predictions under the original
     vs granular gamma curves for the actual March--June 2020 policies
  4. Outputs comparison tables and plots

Usage:
    python sensitivity_granular_policies.py
    python sensitivity_granular_policies.py --regions DE ES BR US-NY
"""
import argparse
import json
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from pandemic_functions.delphi_functions.DELPHI_model import model_covid
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    past_parameters,
    read_oxford_country_policy_data,
    read_policy_data_us_only,
    get_region_gammas_v2,
    run_delphi_policy_scenario,
)
from pandemic_functions.delphi_functions.DELPHI_utils import (
    gamma_t,
    get_initial_conditions,
    create_datasets_with_confidence_intervals,
)
from pandemic_functions.pandemic_params import (
    default_dict_normalized_policy_gamma,
    future_policies,
    global_populations,
    raw_measures,
    region_symbol_country_dict,
    region_symbol_continent_dict,
    p_d, p_h, p_v,
    policy_data_start_date,
    policy_data_end_date,
    bounds_q,
    validcases_threshold_policy,
)
from policy_functions.policy import Policy
from pandemic_functions.pandemic import Pandemic
from pandemic_functions.pandemic_cost import PandemicCost


# ── Granular policy definitions ──────────────────────────────────────────

# Binary features kept separate (after OxCGRT thresholding):
#   C1 = School closing
#   C3 = Cancel public events
#   C4 = Restrictions on gatherings
#   C5 = Close public transport
#   C2 = Workplace closing
#   C7 = Internal movement restrictions
#   C8 = International travel controls
#   C6 = Stay-at-home / Lockdown
#
# Granular MECE categories: each of the original 7 categories is preserved,
# and the two broadest categories---Restrict Mass Gatherings, Schools and
# Others, and Lockdown---are further split by overall OxCGRT stringency
# (sum of C1--C8 raw severity levels) into Moderate vs Strict sub-categories
# using the global median stringency within each category as the cutoff.
# These two categories cover the widest range of policy severity and
# contain the majority of region-days in the data, making them the natural
# candidates for further subdivision.  The remaining five categories each
# represent a more homogeneous set of measures, making further subdivision
# less informative.  Total: 5 unsplit + 2 x 2 = 9 categories.

# Global median stringency per split category (from 210 regions,
# March--July 2020).
STRINGENCY_CUTOFFS = {
    "Restrict_Mass_Gatherings_and_Schools_and_Others": 16,
    "Lockdown": 20,
}

GRANULAR_POLICIES = [
    "G01_No_Measure",
    "G02_Restrict_Mass_Gatherings",
    "G03_Restrict_MG_and_Schools",
    "G04_Others_Only",
    "G05_MG_and_Others",
    "G06_MG_Schools_Others_Moderate",
    "G07_MG_Schools_Others_Strict",
    "G08_Lockdown_Moderate",
    "G09_Lockdown_Strict",
]

GRANULAR_TO_ORIGINAL = {
    "G01_No_Measure": "No_Measure",
    "G02_Restrict_Mass_Gatherings": "Restrict_Mass_Gatherings",
    "G03_Restrict_MG_and_Schools": "Restrict_Mass_Gatherings_and_Schools",
    "G04_Others_Only": "Mass_Gatherings_Authorized_But_Others_Restricted",
    "G05_MG_and_Others": "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "G06_MG_Schools_Others_Moderate": "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "G07_MG_Schools_Others_Strict": "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "G08_Lockdown_Moderate": "Lockdown",
    "G09_Lockdown_Strict": "Lockdown",
}


PARAM_COLS = [
    "Data Start Date",
    "Median Day of Action",
    "Rate of Action",
    "Jump Magnitude",
    "Jump Time",
    "Jump Decay",
]

DEFAULT_REGIONS = ["DE", "ES", "BR", "US-NY"]
REGION_DISPLAY = {"DE": "Germany", "ES": "Spain", "BR": "Brazil", "US-NY": "New York"}


# ── Granular policy data reader (international) ─────────────────────────

_RAW_STRINGENCY_COLS = [
    "C1M_School closing",
    "C2M_Workplace closing",
    "C3M_Cancel public events",
    "C4M_Restrictions on gatherings",
    "C5M_Close public transport",
    "C6M_Stay at home requirements",
    "C7M_Restrictions on internal movement",
    "C8EV_International travel controls",
]

# Maps an original 7-category label to its (moderate, strict) granular pair,
# or to a single granular label if the category is not split.
_ORIG_TO_GRANULAR = {
    "No_Measure": ("G01_No_Measure", None),
    "Restrict_Mass_Gatherings": ("G02_Restrict_Mass_Gatherings", None),
    "Restrict_Mass_Gatherings_and_Schools": ("G03_Restrict_MG_and_Schools", None),
    "Mass_Gatherings_Authorized_But_Others_Restricted": ("G04_Others_Only", None),
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others": (
        "G05_MG_and_Others", None,
    ),
    "Restrict_Mass_Gatherings_and_Schools_and_Others": (
        "G06_MG_Schools_Others_Moderate", "G07_MG_Schools_Others_Strict",
    ),
    "Lockdown": ("G08_Lockdown_Moderate", "G09_Lockdown_Strict"),
}


def read_granular_policy_data_international(
    country: str, start_date: str, end_date: str,
) -> Optional[pd.DataFrame]:
    """
    Read OxCGRT data and assign granular MECE labels (international only).

    Approach: first obtain the original 7-category assignment via the
    production code (read_oxford_country_policy_data), then compute the raw
    OxCGRT stringency score (sum of C1--C8 severity levels, ignoring flags)
    and split each splittable category into Moderate vs Strict using the
    global-median cutoff from STRINGENCY_CUTOFFS.
    """
    orig_df = read_oxford_country_policy_data(
        country=country, start_date=start_date, end_date=end_date,
    )
    if orig_df is None or len(orig_df) == 0:
        return None

    country_rename = {
        "US": "United States",
        "Korea, South": "South Korea",
        "Congo (Kinshasa)": "Democratic Republic of Congo",
        "Czechia": "Czech Republic",
        "Slovakia": "Slovak Republic",
    }
    cname = country_rename.get(country, country)
    raw = raw_measures.copy()
    raw = raw[raw.CountryName == cname].copy()
    if len(raw) == 0:
        return None

    raw["Date"] = raw["Date"].apply(
        lambda x: datetime.strptime(str(x), "%Y%m%d")
    )
    for col in _RAW_STRINGENCY_COLS:
        if col in raw.columns:
            raw[col] = raw.groupby("CountryName")[col].ffill()
    raw["_stringency"] = sum(
        raw[c].fillna(0) for c in _RAW_STRINGENCY_COLS
    )
    raw.rename(columns={"Date": "date"}, inplace=True)
    merged = orig_df.merge(
        raw[["date", "_stringency"]], on="date", how="left",
    )
    merged["_stringency"] = merged["_stringency"].fillna(0)

    output = merged[["country", "province", "date", "_stringency"]].copy()
    for g in GRANULAR_POLICIES:
        output[g] = 0

    for idx in merged.index:
        orig_cat = None
        for p in future_policies:
            if merged.loc[idx, p] == 1:
                orig_cat = p
                break
        if orig_cat is None:
            continue

        s = float(merged.loc[idx, "_stringency"])
        mod_label, strict_label = _ORIG_TO_GRANULAR[orig_cat]

        if strict_label is None:
            label = mod_label
        else:
            cutoff = STRINGENCY_CUTOFFS[orig_cat]
            label = mod_label if s <= cutoff else strict_label

        output.loc[idx, label] = 1

    output = output[
        (output.date >= start_date) & (output.date <= end_date)
    ].reset_index(drop=True)
    return output if len(output) > 0 else None


def read_granular_policy_data(
    country: str, province: str,
    start_date: str, end_date: str,
) -> Optional[pd.DataFrame]:
    """
    Dispatch to granular reader (international) or fall back to original
    7-category reader for US states (mapping back via GRANULAR_TO_ORIGINAL).
    """
    if country == "US" and province != "None":
        df_orig = read_policy_data_us_only(
            state=province, start_date=start_date, end_date=end_date,
        )
        if df_orig is None or len(df_orig) == 0:
            return None

        # Get national US OxCGRT stringency for the Moderate/Strict split
        raw = raw_measures.copy()
        raw = raw[raw.CountryName == "United States"].copy()
        raw["Date"] = raw["Date"].apply(
            lambda x: datetime.strptime(str(x), "%Y%m%d")
        )
        for col in _RAW_STRINGENCY_COLS:
            if col in raw.columns:
                raw[col] = raw.groupby("CountryName")[col].ffill()
        raw["_stringency"] = sum(
            raw[c].fillna(0) for c in _RAW_STRINGENCY_COLS
        )
        raw.rename(columns={"Date": "date"}, inplace=True)
        merged = df_orig.merge(
            raw[["date", "_stringency"]], on="date", how="left",
        )
        merged["_stringency"] = merged["_stringency"].fillna(0)

        df_gran = merged[["country", "province", "date", "_stringency"]].copy()
        for g in GRANULAR_POLICIES:
            df_gran[g] = 0

        for idx in merged.index:
            orig_cat = None
            for p in future_policies:
                if merged.loc[idx, p] == 1:
                    orig_cat = p
                    break
            if orig_cat is None:
                continue
            s = float(merged.loc[idx, "_stringency"])
            mod_label, strict_label = _ORIG_TO_GRANULAR[orig_cat]
            if strict_label is None:
                label = mod_label
            else:
                cutoff = STRINGENCY_CUTOFFS[orig_cat]
                label = mod_label if s <= cutoff else strict_label
            df_gran.loc[idx, label] = 1

        return df_gran
    else:
        return read_granular_policy_data_international(
            country=country, start_date=start_date, end_date=end_date,
        )


# ── Build granular gamma matrix ─────────────────────────────────────────

def build_granular_gamma_matrix(
    start_date: str = "2020-03-15",
    end_date: str = "2020-06-15",
    min_policy_days: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict]:
    """
    Build the partially-observed gamma matrix for the granular partition.
    """
    params_unique = (
        past_parameters
        .sort_values(["Country", "Province", "Data Start Date"])
        .drop_duplicates(subset=["Country", "Province"], keep="last")
        .reset_index(drop=True)
    )

    region_ids: List[str] = []
    gamma_rows: List[np.ndarray] = []
    obs_rows: List[np.ndarray] = []
    n_policies = len(GRANULAR_POLICIES)
    policy_index = {p: i for i, p in enumerate(GRANULAR_POLICIES)}
    global_counts = {p: 0 for p in GRANULAR_POLICIES}

    for _, prow in params_unique.iterrows():
        country = str(prow["Country"])
        province = str(prow["Province"])
        params_list = prow[PARAM_COLS]

        try:
            policy_data = read_granular_policy_data(
                country, province, start_date, end_date,
            )
        except Exception:
            continue
        if policy_data is None or len(policy_data) == 0:
            continue

        policy_gamma_sum = {p: 0.0 for p in GRANULAR_POLICIES}
        policy_gamma_count = {p: 0 for p in GRANULAR_POLICIES}

        for _, row in policy_data.iterrows():
            policy_name = None
            for p in GRANULAR_POLICIES:
                if row.get(p, 0) == 1:
                    policy_name = p
                    break
            if policy_name is None:
                continue
            g_val = float(gamma_t(row["date"], params_list))
            policy_gamma_sum[policy_name] += g_val
            policy_gamma_count[policy_name] += 1

        row_gamma = np.full(n_policies, np.nan)
        row_obs = np.zeros(n_policies, dtype=bool)
        has_any = False
        for p in GRANULAR_POLICIES:
            i = policy_index[p]
            if policy_gamma_count[p] >= min_policy_days:
                row_gamma[i] = policy_gamma_sum[p] / policy_gamma_count[p]
                row_obs[i] = True
                has_any = True
                global_counts[p] += 1

        if not has_any:
            continue

        rid = f"{country}__{province}".replace(" ", "_")
        region_ids.append(rid)
        gamma_rows.append(row_gamma)
        obs_rows.append(row_obs)

    gamma_matrix = np.array(gamma_rows)
    obs_mask = np.array(obs_rows)
    return gamma_matrix, obs_mask, region_ids, GRANULAR_POLICIES, global_counts


# ── ALS rank-1 imputation (new flexible gamma procedure) ─────────────────

def als_rank1_imputation(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    default_g: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank-1 model: gamma_{R,i} = k_R * g_i.

    Both k_R and g_i are estimated from observed entries via alternating
    least squares with non-negativity constraints.  g_i is initialized
    with column means (and optionally with a provided default); both
    initializations are tried and the one with lower residual is kept.
    """
    n_regions, n_policies = gamma_matrix.shape

    def _als_run(g_init):
        g = g_init.copy()
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
                num = gamma_matrix[obs_j, j] @ k[obs_j]
                den = k[obs_j] @ k[obs_j]
                g[j] = max(num / den, 0.0) if den > 0 else g[j]
            change = np.sqrt(np.sum((k - old_k) ** 2)
                             + np.sum((g - old_g) ** 2))
            if change < tol:
                break
        comp = np.outer(k, g)
        np.clip(comp, 0, None, out=comp)
        resid = np.sum((comp[obs_mask] - gamma_matrix[obs_mask]) ** 2)
        comp[obs_mask] = gamma_matrix[obs_mask]
        return comp, k, resid

    col_means = np.zeros(n_policies)
    for j in range(n_policies):
        oj = obs_mask[:, j]
        col_means[j] = gamma_matrix[oj, j].mean() if oj.any() else 0.5

    comp_cm, k_cm, res_cm = _als_run(col_means)

    if default_g is not None:
        comp_def, k_def, res_def = _als_run(default_g)
        if res_def <= res_cm:
            return comp_def, k_def

    return comp_cm, k_cm


def rank_r_svd_imputation(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    rank: int,
    warm_start: Optional[np.ndarray] = None,
    perturb_scale: float = 0.0,
    max_iter: int = 500,
    tol: float = 1e-9,
    seed: int = 42,
) -> np.ndarray:
    """
    Rank-r imputation via truncated SVD (EM-style) with non-negative
    projection, warm-started from a rank-1 ALS solution.
    """
    rng = np.random.default_rng(seed)
    if warm_start is not None:
        X = warm_start.copy()
    else:
        X, _ = als_rank1_imputation(gamma_matrix, obs_mask)
    X[obs_mask] = gamma_matrix[obs_mask]

    if perturb_scale > 0:
        for j in range(gamma_matrix.shape[1]):
            miss_j = ~obs_mask[:, j]
            if miss_j.any():
                std_j = X[miss_j, j].std() if miss_j.sum() > 1 else 0.1
                X[miss_j, j] += rng.normal(0, perturb_scale * std_j,
                                           size=miss_j.sum())
                np.clip(X[miss_j, j], 0, None, out=X[miss_j, j])

    for _ in range(max_iter):
        np.clip(X, 0.0, None, out=X)
        X[obs_mask] = gamma_matrix[obs_mask]

        col_mean = X.mean(axis=0, keepdims=True)
        X_c = X - col_mean
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        S_trunc = S.copy()
        S_trunc[rank:] = 0.0
        X_approx = col_mean + U @ np.diag(S_trunc) @ Vt

        X_new = X.copy()
        X_new[~obs_mask] = X_approx[~obs_mask]

        change = np.linalg.norm(X_new - X) / (np.linalg.norm(X) + 1e-12)
        X = X_new
        if change < tol:
            break

    np.clip(X, 0.0, None, out=X)
    X[obs_mask] = gamma_matrix[obs_mask]
    return X


# ── Rank-l holdout comparison ────────────────────────────────────────────

def holdout_rmse_comparison(
    gamma_matrix, obs_mask,
    ranks=(1, 2, 3),
    holdout_frac=0.20,
    n_splits=20,
    seed=42,
):
    """Compare holdout RMSE for different rank models using ALS for rank-1."""
    rng = np.random.default_rng(seed)
    obs_indices = np.argwhere(obs_mask)
    n_obs = len(obs_indices)
    n_test = max(1, int(n_obs * holdout_frac))

    results = {r: [] for r in ranks}

    for split in range(n_splits):
        perm = rng.permutation(n_obs)
        test_idx = obs_indices[perm[:n_test]]
        train_mask = obs_mask.copy()
        for r, c in test_idx:
            train_mask[r, c] = False

        true_vals = np.array([gamma_matrix[r, c] for r, c in test_idx])

        r1_completed, _ = als_rank1_imputation(gamma_matrix, train_mask)
        completions = {1: r1_completed}
        for rank in sorted(ranks):
            if rank == 1:
                continue
            ps = 0.1 if rank >= 3 else 0.0
            mi = 2000 if rank >= 3 else 500
            completions[rank] = rank_r_svd_imputation(
                gamma_matrix, train_mask, rank=rank,
                warm_start=r1_completed, perturb_scale=ps,
                max_iter=mi, seed=seed + split * 100 + rank,
            )

        for rank in ranks:
            pred_vals = np.array([completions[rank][r, c] for r, c in test_idx])
            rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
            results[rank].append(rmse)

    return {r: (np.mean(v), np.std(v)) for r, v in results.items()}


# ── Extract gammas for a specific region from completed matrices ──────────

def extract_region_gammas_from_matrix(
    region: str,
    completed_matrix: np.ndarray,
    region_ids: List[str],
    policy_names: List[str],
) -> Dict[str, float]:
    """
    Extract gamma_{R,i} for a specific region from a completed matrix
    produced by ALS rank-1 imputation (new flexible gamma procedure).
    """
    country, province = region_symbol_country_dict[region]
    rid = f"{country}__{province}".replace(" ", "_")
    if rid not in region_ids:
        raise ValueError(f"Region {rid} not found in gamma matrix")
    ri = region_ids.index(rid)
    return {p: float(completed_matrix[ri, j]) for j, p in enumerate(policy_names)}


def get_hybrid_prediction_gammas(
    region: str,
    granular_completed: np.ndarray,
    gran_obs_mask: np.ndarray,
    gran_region_ids: List[str],
    gran_policy_names: List[str],
    orig_completed: np.ndarray,
    orig_obs_mask: np.ndarray,
    orig_region_ids: List[str],
    orig_policy_names: List[str],
) -> Dict[str, float]:
    """
    Build hybrid gammas for predictions: observed granular values where
    available, falling back to the observed parent-category gamma from
    the original matrix for unobserved sub-categories.  ALS-imputed
    values are used only when neither level has observations.
    """
    country, province = region_symbol_country_dict[region]
    rid = f"{country}__{province}".replace(" ", "_")
    gran_i = gran_region_ids.index(rid)
    orig_i = orig_region_ids.index(rid)

    gammas: Dict[str, float] = {}
    for j, p in enumerate(gran_policy_names):
        if gran_obs_mask[gran_i, j]:
            gammas[p] = float(granular_completed[gran_i, j])
        else:
            parent = GRANULAR_TO_ORIGINAL[p]
            parent_j = orig_policy_names.index(parent)
            if orig_obs_mask[orig_i, parent_j]:
                gammas[p] = float(orig_completed[orig_i, parent_j])
            else:
                gammas[p] = float(granular_completed[gran_i, j])
    return gammas


def get_observed_granular_policies(
    region: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    policy_days_thresh: int = 20,
) -> List[str]:
    """Return the list of granular policies observed for this region."""
    country, province = region_symbol_country_dict[region]
    ref_start = start_date or policy_data_start_date
    ref_end = end_date or policy_data_end_date

    policy_data = read_granular_policy_data(
        country, province, ref_start, ref_end,
    )
    if policy_data is None or len(policy_data) == 0:
        return []

    observed = []
    for p in GRANULAR_POLICIES:
        count = int((policy_data[p] == 1).sum())
        if count > policy_days_thresh:
            observed.append(p)
    return observed


# ── Run DELPHI with granular gammas ──────────────────────────────────────

def run_delphi_with_granular_gammas(
    region: str,
    granular_gammas: Dict[str, float],
    start_date: str = "2020-03-15",
    end_date: str = "2020-06-15",
) -> pd.DataFrame:
    """
    Run DELPHI for a region using the actual granular policy sequence and
    granular gamma values.  Returns a prediction DataFrame with daily
    cumulative cases and deaths.
    """
    country, province = region_symbol_country_dict[region]
    continent = region_symbol_continent_dict[region]
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")

    totalcases = pd.read_csv(
        f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
    )

    param_row = past_parameters[
        (past_parameters.Country == country) & (past_parameters.Province == province)
    ].iloc[-1]
    parameter_list = param_row.values.tolist()[5:]
    date_day_since100 = pd.to_datetime(param_row["Data Start Date"])

    validcases = totalcases[
        (totalcases.date >= str(date_day_since100.date()))
        & (totalcases.date <= end_date)
    ][["day_since100", "case_cnt", "death_cnt", "total_hospitalization",
       "people_vaccinated", "people_fully_vaccinated"]].reset_index(drop=True)

    PopulationT = global_populations[
        (global_populations.Country == country) & (global_populations.Province == province)
    ].pop2016.iloc[-1]
    N = PopulationT
    PopulationI = validcases.loc[0, "case_cnt"]
    PopulationD = validcases.loc[0, "death_cnt"]
    PopulationR = (
        validcases.loc[0, "death_cnt"] * 5
        if validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"] > validcases.loc[0, "death_cnt"] * 5
        else 0
    )

    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    maxT = (ed - date_day_since100).days + 1
    policy_startT = (sd - date_day_since100).days + 1

    GLOBAL_PARAMS_FIXED = (N, PopulationR, PopulationD, PopulationI, p_v, p_d, p_h)

    # Build piecewise gamma from granular policy data
    policy_data = read_granular_policy_data(
        country, province, start_date, end_date,
    )
    if policy_data is None or len(policy_data) == 0:
        raise ValueError(f"No granular policy data for {region}")

    # Build daily gamma curve
    daily_gamma = {}
    for _, row in policy_data.iterrows():
        day = pd.to_datetime(row["date"])
        t = (day - date_day_since100).days + 1
        for p in GRANULAR_POLICIES:
            if row.get(p, 0) == 1:
                daily_gamma[t] = granular_gammas.get(p, 1.0)
                break

    # Convert to policy_scenario_gammas format: contiguous windows
    sorted_days = sorted(daily_gamma.keys())
    policy_scenario_gammas = {}
    if sorted_days:
        window_start = sorted_days[0]
        current_gamma = daily_gamma[window_start]
        for i in range(1, len(sorted_days)):
            d = sorted_days[i]
            g = daily_gamma[d]
            if abs(g - current_gamma) > 1e-8 or d != sorted_days[i - 1] + 1:
                policy_scenario_gammas[(window_start, d)] = current_gamma
                window_start = d
                current_gamma = g
        policy_scenario_gammas[(window_start, sorted_days[-1] + 1)] = current_gamma

    x_0_cases = get_initial_conditions(
        params_fitted=parameter_list,
        global_params_fixed=GLOBAL_PARAMS_FIXED,
    )

    t_predictions = list(range(maxT))

    def model_fn(t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay,
                 k1, k2, jump, t_jump, std_normal):
        return model_covid(
            t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay,
            k1, k2, jump, t_jump, std_normal,
            N, policy_scenario_gammas, policy_startT, maxT,
        )

    x_sol = solve_ivp(
        fun=model_fn,
        y0=x_0_cases,
        t_span=[t_predictions[0], t_predictions[-1]],
        t_eval=t_predictions,
        args=tuple(parameter_list),
    ).y

    cases_data_fit = validcases["case_cnt"].tolist()
    deaths_data_fit = validcases["death_cnt"].tolist()
    yesterday = str((sd - timedelta(days=1)).date())
    df_pred, _ = create_datasets_with_confidence_intervals(
        continent, country, province,
        date_day_since100, yesterday, x_sol,
        cases_data_fit, deaths_data_fit, q=bounds_q,
    )
    return df_pred


def run_delphi_with_original_gammas(
    region: str,
    start_date: str = "2020-03-15",
    end_date: str = "2020-06-15",
    orig_gammas: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Run DELPHI for a region using the original 7-category policy mapping
    and provided gamma values. Returns prediction DataFrame.
    """
    country, province = region_symbol_country_dict[region]
    continent = region_symbol_continent_dict[region]
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")

    totalcases = pd.read_csv(
        f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
    )

    param_row = past_parameters[
        (past_parameters.Country == country) & (past_parameters.Province == province)
    ].iloc[-1]
    parameter_list = param_row.values.tolist()[5:]
    date_day_since100 = pd.to_datetime(param_row["Data Start Date"])

    validcases = totalcases[
        (totalcases.date >= str(date_day_since100.date()))
        & (totalcases.date <= end_date)
    ][["day_since100", "case_cnt", "death_cnt", "total_hospitalization",
       "people_vaccinated", "people_fully_vaccinated"]].reset_index(drop=True)

    PopulationT = global_populations[
        (global_populations.Country == country) & (global_populations.Province == province)
    ].pop2016.iloc[-1]
    N = PopulationT
    PopulationI = validcases.loc[0, "case_cnt"]
    PopulationD = validcases.loc[0, "death_cnt"]
    PopulationR = (
        validcases.loc[0, "death_cnt"] * 5
        if validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"] > validcases.loc[0, "death_cnt"] * 5
        else 0
    )

    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    maxT = (ed - date_day_since100).days + 1
    policy_startT = (sd - date_day_since100).days + 1

    GLOBAL_PARAMS_FIXED = (N, PopulationR, PopulationD, PopulationI, p_v, p_d, p_h)

    if orig_gammas is None:
        orig_gammas, _, _ = get_region_gammas_v2(region)

    if country == "US":
        policy_data = read_policy_data_us_only(
            state=province, start_date=start_date, end_date=end_date,
        )
    else:
        policy_data = read_oxford_country_policy_data(
            country=country, start_date=start_date, end_date=end_date,
        )

    daily_gamma = {}
    for _, row in policy_data.iterrows():
        day = pd.to_datetime(row["date"])
        t = (day - date_day_since100).days + 1
        for p in future_policies:
            if row.get(p, 0) == 1:
                daily_gamma[t] = orig_gammas.get(p, 1.0)
                break

    sorted_days = sorted(daily_gamma.keys())
    policy_scenario_gammas = {}
    if sorted_days:
        window_start = sorted_days[0]
        current_gamma = daily_gamma[window_start]
        for i in range(1, len(sorted_days)):
            d = sorted_days[i]
            g = daily_gamma[d]
            if abs(g - current_gamma) > 1e-8 or d != sorted_days[i - 1] + 1:
                policy_scenario_gammas[(window_start, d)] = current_gamma
                window_start = d
                current_gamma = g
        policy_scenario_gammas[(window_start, sorted_days[-1] + 1)] = current_gamma

    x_0_cases = get_initial_conditions(
        params_fitted=parameter_list,
        global_params_fixed=GLOBAL_PARAMS_FIXED,
    )
    t_predictions = list(range(maxT))

    def model_fn(t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay,
                 k1, k2, jump, t_jump, std_normal):
        return model_covid(
            t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay,
            k1, k2, jump, t_jump, std_normal,
            N, policy_scenario_gammas, policy_startT, maxT,
        )

    x_sol = solve_ivp(
        fun=model_fn,
        y0=x_0_cases,
        t_span=[t_predictions[0], t_predictions[-1]],
        t_eval=t_predictions,
        args=tuple(parameter_list),
    ).y

    cases_data_fit = validcases["case_cnt"].tolist()
    deaths_data_fit = validcases["death_cnt"].tolist()
    yesterday = str((sd - timedelta(days=1)).date())
    df_pred, _ = create_datasets_with_confidence_intervals(
        continent, country, province,
        date_day_since100, yesterday, x_sol,
        cases_data_fit, deaths_data_fit, q=bounds_q,
    )
    return df_pred


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_gamma_comparison(
    region: str,
    granular_gammas: Dict[str, float],
    original_gammas: Dict[str, float],
    observed_granular: List[str],
    output_dir: Path,
):
    """Bar chart comparing granular vs original gamma values."""
    fig, ax = plt.subplots(figsize=(14, 5))

    gran_sorted = sorted(granular_gammas.items(), key=lambda x: -x[1])
    labels = [k.split("_", 1)[1] for k, _ in gran_sorted]
    gran_vals = [v for _, v in gran_sorted]

    orig_vals = []
    for k, _ in gran_sorted:
        orig_cat = GRANULAR_TO_ORIGINAL[k]
        orig_vals.append(original_gammas.get(orig_cat, 1.0))

    x = np.arange(len(labels))
    width = 0.35
    n_cats = len(GRANULAR_POLICIES)
    bars1 = ax.bar(x - width / 2, gran_vals, width, label=f"Granular ({n_cats} categories)",
                   color="#2c7fb8", alpha=0.8)
    bars2 = ax.bar(x + width / 2, orig_vals, width, label="Original (7 categories)",
                   color="#d95f02", alpha=0.8)

    for i, (k, _) in enumerate(gran_sorted):
        if k in observed_granular:
            ax.text(x[i] - width / 2, gran_vals[i] + 0.01, "*", ha="center",
                    fontsize=14, fontweight="bold", color="#2c7fb8")

    ax.set_xlabel("Granular Policy Category", fontsize=11)
    ax.set_ylabel("$\\gamma_{R,i}$", fontsize=12)
    ax.set_title(f"Gamma Comparison: Granular vs Original ({REGION_DISPLAY.get(region, region)})",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(gran_vals), max(orig_vals)) * 1.15)
    plt.tight_layout()
    fig.savefig(output_dir / f"gamma_comparison_{region}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"gamma_comparison_{region}.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_comparison(
    region: str,
    pred_granular: pd.DataFrame,
    pred_original: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_dir: Path,
):
    """Side-by-side cases and deaths prediction curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)

    # Filter to policy window
    gran_window = pred_granular[
        (pd.to_datetime(pred_granular["Day"]) >= sd)
        & (pd.to_datetime(pred_granular["Day"]) <= ed)
    ].copy()
    orig_window = pred_original[
        (pd.to_datetime(pred_original["Day"]) >= sd)
        & (pd.to_datetime(pred_original["Day"]) <= ed)
    ].copy()

    if len(gran_window) == 0 or len(orig_window) == 0:
        plt.close(fig)
        return

    gran_days = pd.to_datetime(gran_window["Day"])
    orig_days = pd.to_datetime(orig_window["Day"])

    # Cases
    axes[0].plot(orig_days, orig_window["Total Detected"].values,
                 "-", color="#d95f02", linewidth=2, label="Original (7 cat.)")
    axes[0].plot(gran_days, gran_window["Total Detected"].values,
                 "--", color="#2c7fb8", linewidth=2, label="Granular (9 cat.)")
    axes[0].set_xlabel("Date", fontsize=11)
    axes[0].set_ylabel("Cumulative Detected Cases", fontsize=11)
    axes[0].set_title("Cumulative Cases", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].tick_params(axis="x", rotation=30)

    # Deaths
    axes[1].plot(orig_days, orig_window["Total Detected Deaths"].values,
                 "-", color="#d95f02", linewidth=2, label="Original (7 cat.)")
    axes[1].plot(gran_days, gran_window["Total Detected Deaths"].values,
                 "--", color="#2c7fb8", linewidth=2, label="Granular (9 cat.)")
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].set_ylabel("Cumulative Deaths", fontsize=11)
    axes[1].set_title("Cumulative Deaths", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"DELPHI Predictions: Granular vs Original Policy Partition ({REGION_DISPLAY.get(region, region)})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_dir / f"prediction_comparison_{region}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"prediction_comparison_{region}.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_predictions(
    all_preds: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    start_date: str,
    end_date: str,
    output_dir: Path,
):
    """Combined 2x2 plot of all 4 regions."""
    regions = list(all_preds.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)

    for i, region in enumerate(regions[:4]):
        pred_gran, pred_orig = all_preds[region]
        ax = axes[i]

        gran_w = pred_gran[
            (pd.to_datetime(pred_gran["Day"]) >= sd)
            & (pd.to_datetime(pred_gran["Day"]) <= ed)
        ]
        orig_w = pred_orig[
            (pd.to_datetime(pred_orig["Day"]) >= sd)
            & (pd.to_datetime(pred_orig["Day"]) <= ed)
        ]

        if len(gran_w) == 0 or len(orig_w) == 0:
            continue

        gran_days = pd.to_datetime(gran_w["Day"])
        orig_days = pd.to_datetime(orig_w["Day"])

        ax.plot(orig_days, orig_w["Total Detected"].values,
                "-", color="#d95f02", linewidth=2, label="Original (7 cat.)")
        ax.plot(gran_days, gran_w["Total Detected"].values,
                "--", color="#2c7fb8", linewidth=2, label="Granular (9 cat.)")
        ax.set_title(REGION_DISPLAY.get(region, region), fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Cumulative Cases", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(
        "DELPHI Cumulative Case Predictions: Original vs Granular Policy Partition\n"
        "(March 15 -- June 15, 2020)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "combined_prediction_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "combined_prediction_comparison.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Granular NPI policy sensitivity analysis")
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--start-date", default="2020-03-15")
    parser.add_argument("--end-date", default="2020-06-15")
    parser.add_argument("--output-dir", default="simulation_results/granular_policies")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  GRANULAR POLICY PARTITION SENSITIVITY ANALYSIS")
    print("=" * 70)

    # ── Step 1: Build granular gamma matrix and holdout comparison ────
    print("\n[1] Building granular gamma matrix across all regions...")
    gamma_matrix, obs_mask, region_ids, policy_names, global_counts = (
        build_granular_gamma_matrix(
            start_date="2020-03-15", end_date="2020-06-15", min_policy_days=10,
        )
    )
    print(f"    Assembled {gamma_matrix.shape[0]} regions x {gamma_matrix.shape[1]} granular policies")
    print(f"    Observation density: {obs_mask.mean():.1%}")
    print(f"    Policy observation counts:")
    for p, c in sorted(global_counts.items()):
        print(f"      {p}: {c} regions")

    # Save gamma matrix summary
    gamma_df = pd.DataFrame(gamma_matrix, index=region_ids, columns=policy_names)
    gamma_df.to_csv(output_dir / "granular_gamma_matrix.csv")

    # ── Step 2: Holdout RMSE comparison ──────────────────────────────
    print("\n[2] Running holdout RMSE comparison (rank 1 vs 2 vs 3)...")
    rmse_results = holdout_rmse_comparison(gamma_matrix, obs_mask, ranks=(1, 2, 3))
    print("    Holdout RMSE (mean +/- std over 20 splits, 20% held out):")
    rmse_table = {}
    for rank, (mean_rmse, std_rmse) in rmse_results.items():
        print(f"      Rank {rank}: {mean_rmse:.4f} +/- {std_rmse:.4f}")
        rmse_table[f"rank_{rank}"] = {"mean": round(mean_rmse, 4), "std": round(std_rmse, 4)}

    # Also build original 7-category matrix for comparison
    # Use the same holdout function as the first-point analysis for consistency
    print("\n[2b] Building original 7-category gamma matrix for comparison...")
    from analyze_gamma_rank import build_gamma_matrix as build_original_gamma_matrix
    from run_multiperiod_lowrank import holdout_multirank
    orig_gamma_matrix, orig_obs_mask, orig_region_ids, orig_policy_names = (
        build_original_gamma_matrix(
            start_date="2020-03-15", end_date="2020-06-15", min_policy_days=10,
        )
    )
    orig_rmse_raw = holdout_multirank(
        orig_gamma_matrix, orig_obs_mask, ranks=[1, 2, 3],
        n_splits=20, holdout_frac=0.2, seed=42,
    )
    print("    Original 7-category holdout RMSE:")
    orig_rmse_table = {}
    for rank in [1, 2, 3]:
        mean_rmse, std_rmse, _ = orig_rmse_raw[rank]
        print(f"      Rank {rank}: {mean_rmse:.4f} +/- {std_rmse:.4f}")
        orig_rmse_table[f"rank_{rank}"] = {"mean": round(mean_rmse, 4), "std": round(std_rmse, 4)}

    # ── Step 2c: ALS rank-1 completed matrices for predictions ─────
    print("\n[2c] Computing ALS rank-1 completed matrices...")
    granular_completed, _ = als_rank1_imputation(gamma_matrix, obs_mask)
    print(f"    Granular completed matrix: {granular_completed.shape}")

    from analyze_gamma_rank import rank1_imputation as orig_rank1_imputation
    orig_completed, _ = orig_rank1_imputation(orig_gamma_matrix, orig_obs_mask)
    print(f"    Original completed matrix: {orig_completed.shape}")

    # ── Step 3: Per-region predictions using ALS-completed gammas ──
    print(f"\n[3] Running predictions for regions: {args.regions}")
    all_predictions = {}
    comparison_table = []

    for region in args.regions:
        print(f"\n  --- {REGION_DISPLAY.get(region, region)} ({region}) ---")
        try:
            granular_gammas = get_hybrid_prediction_gammas(
                region,
                granular_completed, obs_mask, region_ids, policy_names,
                orig_completed, orig_obs_mask, orig_region_ids, orig_policy_names,
            )
            original_gammas = extract_region_gammas_from_matrix(
                region, orig_completed, orig_region_ids, orig_policy_names,
            )
            observed_granular = get_observed_granular_policies(
                region, start_date="2020-03-15", end_date="2020-06-15",
            )

            print(f"    Observed granular policies: {observed_granular}")
            print(f"    Granular gammas (hybrid: observed + parent fallback):")
            for p, g in sorted(granular_gammas.items()):
                marker = "*" if p in observed_granular else " "
                orig_cat = GRANULAR_TO_ORIGINAL[p]
                orig_g = original_gammas.get(orig_cat, float("nan"))
                print(f"      {marker} {p}: {g:.4f}  (orig {orig_cat}: {orig_g:.4f})")

            plot_gamma_comparison(
                region, granular_gammas, original_gammas,
                observed_granular, output_dir,
            )

            print(f"    Running DELPHI with granular gammas (hybrid)...")
            pred_granular = run_delphi_with_granular_gammas(
                region, granular_gammas,
                start_date=args.start_date, end_date=args.end_date,
            )
            print(f"    Running DELPHI with original gammas (ALS)...")
            pred_original = run_delphi_with_original_gammas(
                region, start_date=args.start_date, end_date=args.end_date,
                orig_gammas=original_gammas,
            )

            all_predictions[region] = (pred_granular, pred_original)

            # Plot individual comparison
            plot_prediction_comparison(
                region, pred_granular, pred_original,
                args.start_date, args.end_date, output_dir,
            )

            # Compute summary statistics
            sd = pd.to_datetime(args.start_date)
            ed = pd.to_datetime(args.end_date)
            gran_w = pred_granular[
                (pd.to_datetime(pred_granular["Day"]) >= sd)
                & (pd.to_datetime(pred_granular["Day"]) <= ed)
            ]
            orig_w = pred_original[
                (pd.to_datetime(pred_original["Day"]) >= sd)
                & (pd.to_datetime(pred_original["Day"]) <= ed)
            ]

            if len(gran_w) > 0 and len(orig_w) > 0:
                gran_cases = gran_w["Total Detected"].values[-1] - gran_w["Total Detected"].values[0]
                orig_cases = orig_w["Total Detected"].values[-1] - orig_w["Total Detected"].values[0]
                gran_deaths = gran_w["Total Detected Deaths"].values[-1] - gran_w["Total Detected Deaths"].values[0]
                orig_deaths = orig_w["Total Detected Deaths"].values[-1] - orig_w["Total Detected Deaths"].values[0]
                case_pct_diff = abs(gran_cases - orig_cases) / max(orig_cases, 1) * 100
                death_pct_diff = abs(gran_deaths - orig_deaths) / max(orig_deaths, 1) * 100

                comparison_table.append({
                    "region": region,
                    "display_name": REGION_DISPLAY.get(region, region),
                    "orig_cases": int(orig_cases),
                    "gran_cases": int(gran_cases),
                    "case_pct_diff": round(case_pct_diff, 2),
                    "orig_deaths": int(orig_deaths),
                    "gran_deaths": int(gran_deaths),
                    "death_pct_diff": round(death_pct_diff, 2),
                })
                print(f"    Cases:  original={int(orig_cases):,}, granular={int(gran_cases):,} "
                      f"(diff={case_pct_diff:.1f}%)")
                print(f"    Deaths: original={int(orig_deaths):,}, granular={int(gran_deaths):,} "
                      f"(diff={death_pct_diff:.1f}%)")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── Step 4: Combined plot ────────────────────────────────────────
    if len(all_predictions) >= 2:
        plot_combined_predictions(
            all_predictions, args.start_date, args.end_date, output_dir,
        )

    # ── Step 5: Save summary ─────────────────────────────────────────
    comp_df = pd.DataFrame(comparison_table)
    comp_df.to_csv(output_dir / "prediction_comparison_table.csv", index=False)

    summary = {
        "analysis": "granular_policy_sensitivity",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_granular_policies": len(GRANULAR_POLICIES),
        "granular_policies": GRANULAR_POLICIES,
        "granular_to_original_mapping": GRANULAR_TO_ORIGINAL,
        "gamma_matrix_shape": list(gamma_matrix.shape),
        "observation_density": round(float(obs_mask.mean()), 4),
        "global_policy_counts": global_counts,
        "holdout_rmse_granular": rmse_table,
        "holdout_rmse_original": orig_rmse_table,
        "prediction_comparison": comparison_table,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  All outputs saved to: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
