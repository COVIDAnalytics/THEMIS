import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from pandemic_functions.delphi_functions.DELPHI_model import model_covid as model_covid_v2
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    read_oxford_country_policy_data,
    read_policy_data_us_only,
)
from pandemic_functions.delphi_functions.DELPHI_utils import (
    gamma_t as gamma_t_v2,
    get_initial_conditions as get_initial_conditions_v2,
)
from pandemic_functions.pandemic_params import (
    DetectD,
    IncubeD,
    RecoverHD,
    RecoverID,
    VentilatedD,
    default_dict_normalized_policy_gamma,
    future_policies,
    global_populations,
    p_d,
    p_h,
    p_v,
)


DEFAULT_TRUTH_PATH = "pandemic_functions/pandemic_data/Global_DELPHI_predictions_combined.csv"
DEFAULT_OUTPUT_DIR = "simulation_results"
DEFAULT_RATE_OF_DEATH_V1 = 0.2
MAX_TRAIN_FRACTION = 0.8

MODEL_ORDER = ["themis_transfer", "delphi_full", "delphi_constant", "seir"]
MODEL_LABELS = {
    "themis_transfer": "THEMIS-transfer",
    "delphi_full": "DELPHI-full",
    "delphi_constant": "DELPHI-constant",
    "seir": "SEIR",
}
MODEL_COLORS = {
    "themis_transfer": "#1f77b4",
    "delphi_full": "#2ca02c",
    "delphi_constant": "#d62728",
    "seir": "#7f7f7f",
}


@dataclass(frozen=True)
class CutoffConfig:
    label: str
    cutoff_date: str
    param_file: str
    model_version: str
    global_prediction_file: str


CUTOFF_CONFIGS: Dict[str, CutoffConfig] = {
    # 0415 snapshot is stored as 0414 in the data folder.
    "20200415": CutoffConfig(
        label="20200415",
        cutoff_date="2020-04-14",
        param_file="pandemic_functions/pandemic_data/Parameters_Global_20200414.csv",
        model_version="v1",
        global_prediction_file="pandemic_functions/pandemic_data/Global_20200414.csv",
    ),
    "20200515": CutoffConfig(
        label="20200515",
        cutoff_date="2020-05-15",
        param_file="pandemic_functions/pandemic_data/Parameters_Global_20200515.csv",
        model_version="v1",
        global_prediction_file="pandemic_functions/pandemic_data/Global_20200515.csv",
    ),
    "20200615": CutoffConfig(
        label="20200615",
        cutoff_date="2020-06-15",
        param_file="pandemic_functions/pandemic_data/Parameters_Global_20200615.csv",
        model_version="v1",
        global_prediction_file="pandemic_functions/pandemic_data/Global_20200615.csv",
    ),
    "20200715": CutoffConfig(
        label="20200715",
        cutoff_date="2020-07-15",
        param_file="pandemic_functions/pandemic_data/Parameters_Global_V2_20200715.csv",
        model_version="v2",
        global_prediction_file="pandemic_functions/pandemic_data/Global_V2_20200715.csv",
    ),
}


@dataclass
class RegionArtifact:
    region_id: str
    continent: str
    country: str
    province: str
    cutoff_label: str
    cutoff_date: datetime
    data_start_date: datetime
    model_version: str
    delphi_params: tuple
    population: float
    initial_cases: float
    initial_deaths: float
    policy_at_cutoff: str
    policy_vector: List[str]
    actual_cases_by_horizon: Dict[int, float]
    actual_deaths_by_horizon: Dict[int, float]
    gamma_sum_precutoff: Dict[str, float]
    gamma_count_precutoff: Dict[str, int]
    gamma_sum_by_horizon: Dict[int, Dict[str, float]]
    gamma_count_by_horizon: Dict[int, Dict[str, int]]
    z_sum_by_horizon: Dict[int, float]
    z_count_by_horizon: Dict[int, int]
    region_truth: pd.DataFrame
    baseline_predictions: Dict[str, Dict[int, Tuple[float, float]]]


_POLICY_CACHE: Dict[Tuple[str, str, str, str], Optional[pd.DataFrame]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-date THEMIS transferability benchmark against DELPHI-full, "
            "DELPHI-constant, and SEIR baselines."
        )
    )
    parser.add_argument(
        "--cutoffs",
        type=str,
        default="20200415,20200515,20200615,20200715",
        help="Comma-separated cutoff labels (subset of 20200415,20200515,20200615,20200715).",
    )
    parser.add_argument("--max-months", type=int, default=3, help="Prediction horizons in months (1..max-months).")
    parser.add_argument("--n-splits", type=int, default=20, help="Number of random train/test splits per cutoff.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Train-region fraction (must be <= 0.8).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-regions", type=int, default=0, help="Optional cap for fast debugging.")
    parser.add_argument(
        "--min-precutoff-days",
        type=int,
        default=14,
        help="Minimum policy/truth days before cutoff to keep a region.",
    )
    parser.add_argument(
        "--min-seir-train-days",
        type=int,
        default=14,
        help="Minimum train days for SEIR fitting.",
    )
    parser.add_argument("--truth-path", type=str, default=DEFAULT_TRUTH_PATH, help="Historical truth CSV path.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Parent output directory.")
    parser.add_argument("--robust-stat", type=str, default="median", choices=["mean", "median", "trimmed_mean"])
    parser.add_argument("--trim-quantile", type=float, default=0.1)
    parser.add_argument("--z-stat", type=str, default="median", choices=["mean", "median"])
    parser.add_argument("--min-policy-regions", type=int, default=2)
    parser.add_argument("--shrinkage-regions", type=float, default=12.0)
    parser.add_argument("--gamma-floor", type=float, default=0.05)
    parser.add_argument("--gamma-cap", type=float, default=1.2)
    parser.add_argument(
        "--gamma-estimator",
        type=str,
        default="paper_day_mean",
        choices=["paper_day_mean", "robust"],
        help=(
            "Estimator for global policy gamma_i from training regions. "
            "'paper_day_mean' matches day-weighted formula in manuscript; "
            "'robust' keeps region-balanced robust estimator."
        ),
    )
    parser.add_argument(
        "--transfer-gamma-mode",
        type=str,
        default="relative_to_cutoff",
        choices=["absolute", "relative_to_cutoff"],
        help=(
            "How to apply train-estimated policy gammas on test regions. "
            "'absolute' uses direct transfer; "
            "'relative_to_cutoff' transfers policy ratios anchored at each region cutoff gamma."
        ),
    )
    parser.add_argument(
        "--k-estimation-mode",
        type=str,
        default="precutoff_weighted",
        choices=["precutoff_weighted", "cutoff_policy"],
        help=(
            "How to estimate region-specific k_R when transfer-gamma-mode is relative_to_cutoff. "
            "'precutoff_weighted' fits k_R from region's own pre-cutoff policy means; "
            "'cutoff_policy' anchors k_R to policy active at cutoff only."
        ),
    )
    parser.add_argument(
        "--k-floor",
        type=float,
        default=0.0,
        help="Lower clip for k_R after estimation.",
    )
    parser.add_argument(
        "--k-cap",
        type=float,
        default=10.0,
        help="Upper clip for k_R after estimation.",
    )
    parser.add_argument(
        "--k-shrinkage-to-one",
        type=float,
        default=0.0,
        help="Optional shrinkage of k_R toward 1.0 (in [0,1]).",
    )
    parser.add_argument(
        "--relative-gamma-floor",
        type=float,
        default=0.0,
        help="Floor used after relative-to-cutoff gamma transfer.",
    )
    parser.add_argument(
        "--relative-gamma-cap",
        type=float,
        default=1.2,
        help="Cap used after relative-to-cutoff gamma transfer.",
    )
    parser.add_argument(
        "--min-eval-actual-cases",
        type=float,
        default=0.0,
        help="Exclude rows with actual cases below this threshold from APE aggregation.",
    )
    parser.add_argument(
        "--min-eval-actual-deaths",
        type=float,
        default=0.0,
        help="Exclude rows with actual deaths below this threshold from APE aggregation.",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Primary APE metric for significance tests and headline plots.",
    )
    return parser.parse_args()


def _to_date(x: str) -> datetime:
    return pd.to_datetime(x).normalize().to_pydatetime()


def _safe_ape(pred: float, actual: float) -> float:
    if actual <= 0:
        return np.nan
    return abs(pred - actual) / abs(actual) * 100.0


def _region_id(country: str, province: str) -> str:
    return f"{country}__{province}".replace(" ", "_")


def _to_float(x, default: float = np.nan) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _active_policy_from_row(row: pd.Series) -> Optional[str]:
    for policy in future_policies:
        if int(row.get(policy, 0)) == 1:
            return policy
    return None


def _dominant_policy(policy_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> Optional[str]:
    sub = policy_df[(policy_df["date"] >= start_date) & (policy_df["date"] <= end_date)]
    if sub.empty:
        return None
    counts = [int(sub[p].sum()) for p in future_policies]
    expanded: List[int] = []
    for idx, c in enumerate(counts):
        expanded.extend([idx] * c)
    if len(expanded) == 0:
        return None
    return future_policies[int(np.median(expanded))]


def _load_truth(truth_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        truth_path,
        keep_default_na=False,
        usecols=["Country", "Province", "Day", "Total Detected True", "Total Detected Deaths True"],
    )
    df["Country"] = df["Country"].astype(str)
    df["Province"] = df["Province"].fillna("None").astype(str)
    df["date"] = pd.to_datetime(df["Day"], errors="coerce").dt.normalize()
    df["case_cnt"] = pd.to_numeric(df["Total Detected True"], errors="coerce")
    df["death_cnt"] = pd.to_numeric(df["Total Detected Deaths True"], errors="coerce")
    df = df.dropna(subset=["date", "case_cnt", "death_cnt"]).copy()
    df = df.rename(columns={"Country": "country", "Province": "province"})
    return df[["country", "province", "date", "case_cnt", "death_cnt"]].sort_values(
        ["country", "province", "date"]
    )


def _build_population_lookup() -> Dict[Tuple[str, str], float]:
    df = global_populations.copy()
    df["Country"] = df["Country"].astype(str)
    df["Province"] = df["Province"].fillna("None").astype(str)
    df["pop2016"] = pd.to_numeric(df["pop2016"], errors="coerce")
    df = df.dropna(subset=["pop2016"]).copy()
    df = df.sort_values(["Country", "Province", "pop2016"]).drop_duplicates(
        subset=["Country", "Province"], keep="last"
    )
    return {(str(r.Country), str(r.Province)): float(r.pop2016) for _, r in df.iterrows()}


def _read_policy_data_cached(country: str, province: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    key = (country, province, start_date, end_date)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    try:
        if country == "US":
            df = read_policy_data_us_only(state=province, start_date=start_date, end_date=end_date)
        else:
            df = read_oxford_country_policy_data(country=country, start_date=start_date, end_date=end_date)
    except Exception:
        _POLICY_CACHE[key] = None
        return None
    if df is None or len(df) == 0:
        _POLICY_CACHE[key] = None
        return None
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    for p in future_policies:
        if p not in out.columns:
            out[p] = 0
        out[p] = out[p].fillna(0).astype(int)
    _POLICY_CACHE[key] = out[["date"] + future_policies].reset_index(drop=True)
    return _POLICY_CACHE[key]


def _parse_delphi_params(row: pd.Series, model_version: str) -> tuple:
    alpha = _to_float(row.get("Infection Rate"), 0.0)
    days = _to_float(row.get("Median Day of Action"), 0.0)
    r_s = _to_float(row.get("Rate of Action"), 0.0)
    if model_version == "v2":
        return (
            alpha,
            days,
            r_s,
            _to_float(row.get("Rate of Death"), 0.05),
            _to_float(row.get("Mortality Rate"), 0.05),
            _to_float(row.get("Rate of Mortality Rate Decay"), 0.0),
            _to_float(row.get("Internal Parameter 1"), 1.0),
            _to_float(row.get("Internal Parameter 2"), 1.0),
            _to_float(row.get("Jump Magnitude"), 0.0),
            _to_float(row.get("Jump Time"), 80.0),
            _to_float(row.get("Jump Decay"), 1.0),
        )
    return (
        alpha,
        days,
        r_s,
        _to_float(row.get("Rate of Death"), DEFAULT_RATE_OF_DEATH_V1),
        _to_float(row.get("Mortality Rate"), 0.05),
        _to_float(row.get("Internal Parameter 1"), 1.0),
        _to_float(row.get("Internal Parameter 2"), 1.0),
    )


def _gamma_t_v1(day: datetime, artifact: RegionArtifact) -> float:
    _, median_day, rate_of_action, _, _, _, _ = artifact.delphi_params
    t = (day - artifact.data_start_date).days
    return float((2 / np.pi) * np.arctan(-(t - median_day) / 20.0 * rate_of_action) + 1.0)


def _gamma_t_for_day(day: datetime, artifact: RegionArtifact) -> float:
    if artifact.model_version == "v2":
        alpha, days, r_s, r_dth, p_dth, r_decay, k1, k2, jump, t_jump, std = artifact.delphi_params
        return float(gamma_t_v2(day, [artifact.data_start_date, days, r_s, jump, t_jump, std]))
    return _gamma_t_v1(day, artifact)


def _build_gamma_stats(artifact: RegionArtifact, policy_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, int], float, int]:
    default_gammas = dict(sorted(default_dict_normalized_policy_gamma.items(), key=lambda x: x[0]))
    gamma_sum = {p: 0.0 for p in default_gammas}
    gamma_count = {p: 0 for p in default_gammas}
    z_sum = 0.0
    z_count = 0
    for _, row in policy_df.iterrows():
        policy = _active_policy_from_row(row)
        if policy is None or policy not in default_gammas:
            continue
        g_val = _gamma_t_for_day(row["date"].to_pydatetime(), artifact)
        gamma_sum[policy] += g_val
        gamma_count[policy] += 1
        z_sum += g_val / max(default_gammas[policy], 1e-8)
        z_count += 1
    return gamma_sum, gamma_count, z_sum, z_count


def _initial_conditions_v1(params_fitted: tuple, N: float, pop_ci: float, pop_r: float, pop_d: float, pop_i: float) -> list:
    alpha, days, r_s, r_dth, p_dth, k1, k2 = params_fitted
    S_0 = (N - pop_ci / p_d) - (pop_ci / p_d * (k1 + k2)) - (pop_r / p_d) - (pop_d / p_d)
    E_0 = pop_ci / p_d * k1
    I_0 = pop_ci / p_d * k2
    AR_0 = (pop_ci / p_d - pop_ci) * (1 - p_dth)
    DHR_0 = (pop_ci * p_h) * (1 - p_dth)
    DQR_0 = pop_ci * (1 - p_h) * (1 - p_dth)
    AD_0 = (pop_ci / p_d - pop_ci) * p_dth
    DHD_0 = pop_ci * p_h * p_dth
    DQD_0 = pop_ci * (1 - p_h) * p_dth
    R_0 = pop_r / p_d
    D_0 = pop_d / p_d
    TH_0 = pop_ci * p_h
    DVR_0 = (pop_ci * p_h * p_v) * (1 - p_dth)
    DVD_0 = (pop_ci * p_h * p_v) * p_dth
    DD_0 = pop_d
    DT_0 = pop_i
    return [S_0, E_0, I_0, AR_0, DHR_0, DQR_0, AD_0, DHD_0, DQD_0, R_0, D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0]


def _model_covid_v1(
    t: float,
    x: Sequence[float],
    alpha: float,
    days: float,
    r_s: float,
    r_dth: float,
    p_dth: float,
    k1: float,
    k2: float,
    N: float,
    policy_windows: Optional[Dict[Tuple[float, float], float]],
    policy_start_t: Optional[float],
    max_t: int,
) -> list:
    r_i = np.log(2) / IncubeD
    r_d = np.log(2) / DetectD
    r_ri = np.log(2) / RecoverID
    r_rh = np.log(2) / RecoverHD
    r_rv = np.log(2) / VentilatedD
    gamma_val = (2 / np.pi) * np.arctan(-(t - days) / 20.0 * r_s) + 1.0
    if policy_windows is not None and policy_start_t is not None and t >= policy_start_t and t < max_t:
        for (t1, t2), g in policy_windows.items():
            if t >= t1 and t < t2:
                gamma_val = g
                break
    S, E, I, AR, DHR, DQR, AD, DHD, DQD, R, D, TH, DVR, DVD, DD, DT = x
    dSdt = -alpha * gamma_val * S * I / N
    dEdt = alpha * gamma_val * S * I / N - r_i * E
    dIdt = r_i * E - r_d * I
    dARdt = r_d * (1 - p_dth) * (1 - p_d) * I - r_ri * AR
    dDHRdt = r_d * (1 - p_dth) * p_d * p_h * I - r_rh * DHR
    dDQRdt = r_d * (1 - p_dth) * p_d * (1 - p_h) * I - r_ri * DQR
    dADdt = r_d * p_dth * (1 - p_d) * I - r_dth * AD
    dDHDdt = r_d * p_dth * p_d * p_h * I - r_dth * DHD
    dDQDdt = r_d * p_dth * p_d * (1 - p_h) * I - r_dth * DQD
    dRdt = r_ri * (AR + DQR) + r_rh * DHR
    dDdt = r_dth * (AD + DQD + DHD)
    dTHdt = r_d * p_d * p_h * I
    dDVRdt = r_d * (1 - p_dth) * p_d * p_h * p_v * I - r_rv * DVR
    dDVDdt = r_d * p_dth * p_d * p_h * p_v * I - r_dth * DVD
    dDDdt = r_dth * (DHD + DQD)
    dDTdt = r_d * p_d * I
    return [dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt, dDQDdt, dRdt, dDdt, dTHdt, dDVRdt, dDVDdt, dDDdt, dDTdt]


def _simulate_delphi(
    artifact: RegionArtifact,
    months: int,
    mode: str,
    themis_gammas: Optional[Dict[str, float]] = None,
) -> Optional[Dict[int, Tuple[float, float]]]:
    cutoff = artifact.cutoff_date
    end_date = cutoff + relativedelta(months=months)
    max_day = (end_date - artifact.data_start_date).days
    if max_day <= 1:
        return None
    t_eval = np.arange(0, max_day + 1, dtype=float)
    policy_start_t = float((cutoff - artifact.data_start_date).days)
    policy_windows: Optional[Dict[Tuple[float, float], float]] = None

    if mode == "constant":
        gamma_const = _gamma_t_for_day(cutoff, artifact)
        policy_windows = {(policy_start_t, float(max_day) + 1.0): gamma_const}
    elif mode == "themis_transfer":
        if themis_gammas is None:
            return None
        policy_windows = {}
        for m in range(months):
            start = cutoff + relativedelta(months=m)
            end = cutoff + relativedelta(months=m + 1)
            t1 = float((start - artifact.data_start_date).days)
            t2 = float((end - artifact.data_start_date).days)
            policy_name = artifact.policy_vector[m]
            policy_windows[(t1, t2)] = float(themis_gammas[policy_name])

    if artifact.model_version == "v2":
        pop_i = artifact.initial_cases
        pop_d0 = artifact.initial_deaths
        pop_r = pop_d0 * 5 if pop_i - pop_d0 > pop_d0 * 5 else 0.0
        x0 = get_initial_conditions_v2(
            params_fitted=artifact.delphi_params,
            global_params_fixed=(artifact.population, pop_r, pop_d0, pop_i, p_v, p_d, p_h),
        )

        def rhs(t, x, alpha, days, r_s, r_dth, p_dth, r_decay, k1, k2, jump, t_jump, std):
            return model_covid_v2(
                t,
                x,
                alpha,
                days,
                r_s,
                r_dth,
                p_dth,
                r_decay,
                k1,
                k2,
                jump,
                t_jump,
                std,
                artifact.population,
                policy_windows,
                policy_start_t if mode != "full" else None,
                max_day + 1,
            )

        try:
            sol = solve_ivp(
                fun=rhs,
                y0=x0,
                t_span=[0, float(max_day)],
                t_eval=t_eval,
                args=artifact.delphi_params,
                rtol=1e-6,
                atol=1e-6,
            )
        except Exception:
            return None
    else:
        params_v1 = artifact.delphi_params
        pop_i = artifact.initial_cases
        pop_d0 = artifact.initial_deaths
        pop_r = pop_d0 * 5.0
        pop_ci = pop_i - pop_d0 - pop_r
        if pop_ci <= 0:
            return None
        x0 = _initial_conditions_v1(params_v1, artifact.population, pop_ci, pop_r, pop_d0, pop_i)

        def rhs(t, x, alpha, days, r_s, r_dth, p_dth, k1, k2):
            return _model_covid_v1(
                t,
                x,
                alpha,
                days,
                r_s,
                r_dth,
                p_dth,
                k1,
                k2,
                artifact.population,
                policy_windows,
                policy_start_t if mode != "full" else None,
                max_day + 1,
            )

        try:
            sol = solve_ivp(
                fun=rhs,
                y0=x0,
                t_span=[0, float(max_day)],
                t_eval=t_eval,
                args=params_v1,
                rtol=1e-6,
                atol=1e-6,
            )
        except Exception:
            return None

    if (not sol.success) or (sol.y.shape[1] <= 2):
        return None

    total_cases = sol.y[15, :]
    total_deaths = sol.y[14, :]
    start_idx = int((cutoff - artifact.data_start_date).days)
    if start_idx < 0 or start_idx >= len(total_cases):
        return None

    intervals: Dict[int, Tuple[float, float]] = {}
    for h in range(1, months + 1):
        end_h = cutoff + relativedelta(months=h)
        end_idx = int((end_h - artifact.data_start_date).days)
        if end_idx < 0 or end_idx >= len(total_cases):
            return None
        pred_cases = float(max(total_cases[end_idx] - total_cases[start_idx], 0.0))
        pred_deaths = float(max(total_deaths[end_idx] - total_deaths[start_idx], 0.0))
        intervals[h] = (pred_cases, pred_deaths)
    return intervals


def _fit_predict_seir(artifact: RegionArtifact, months: int, min_train_days: int) -> Optional[Dict[int, Tuple[float, float]]]:
    cutoff = artifact.cutoff_date
    end_date = cutoff + relativedelta(months=months)
    df = artifact.region_truth[
        (artifact.region_truth["date"] >= artifact.data_start_date) & (artifact.region_truth["date"] <= end_date)
    ].copy()
    if df.empty:
        return None
    train = df[df["date"] <= cutoff].copy()
    if train["date"].nunique() < min_train_days:
        return None

    t_train = (train["date"] - artifact.data_start_date).dt.days.to_numpy(dtype=float)
    y_cases = train["case_cnt"].to_numpy(dtype=float)
    y_deaths = train["death_cnt"].to_numpy(dtype=float)
    cases0 = float(y_cases[0])
    deaths0 = float(y_deaths[0])
    max_day = int((end_date - artifact.data_start_date).days)
    t_eval_full = np.arange(0, max_day + 1, dtype=float)
    sigma = 1.0 / 5.2
    gamma_rec = 1.0 / 10.0

    def solve_theta(theta: np.ndarray, t_eval: np.ndarray):
        beta, mu, detect, e0, i0 = theta
        e0 = max(float(e0), 1.0)
        i0 = max(float(i0), 1.0)
        d0 = max(deaths0, 0.0)
        s0 = artifact.population - e0 - i0 - d0
        if s0 <= 1.0:
            return None
        x0 = [s0, e0, i0, 0.0, d0, max(cases0, 1.0)]

        def rhs(t, x, beta_param, mu_param, detect_param):
            S, E, I, R, D, Cdet = x
            new_exposed = beta_param * S * I / artifact.population
            dS = -new_exposed
            dE = new_exposed - sigma * E
            dI = sigma * E - gamma_rec * I - mu_param * I
            dR = gamma_rec * I
            dD = mu_param * I
            dC = detect_param * sigma * E
            return [dS, dE, dI, dR, dD, dC]

        try:
            sol = solve_ivp(
                fun=rhs,
                y0=x0,
                t_span=[0, float(t_eval[-1])],
                t_eval=t_eval,
                args=(beta, mu, detect),
                rtol=1e-5,
                atol=1e-5,
            )
        except Exception:
            return None
        if not sol.success:
            return None
        return sol

    e0_guess = max(cases0 * 0.5, 10.0)
    i0_guess = max(cases0 * 0.25, 10.0)
    init = np.array([0.8, 0.02, 0.25, e0_guess, i0_guess], dtype=float)
    upper_pop = max(artifact.population * 0.2, 1e4)
    bounds = [(0.05, 3.5), (1e-4, 0.4), (0.02, 1.0), (1.0, upper_pop), (1.0, upper_pop)]

    def objective(theta: np.ndarray) -> float:
        sol = solve_theta(theta, t_train)
        if sol is None:
            return 1e12
        pred_cases = sol.y[5, :]
        pred_deaths = sol.y[4, :]
        w = np.linspace(1.0, 2.0, len(t_train))
        err_cases = (np.log1p(pred_cases) - np.log1p(y_cases)) ** 2
        err_deaths = (np.log1p(pred_deaths) - np.log1p(y_deaths)) ** 2
        return float(np.mean(w * (err_cases + err_deaths)))

    res = minimize(
        objective,
        x0=init,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 120, "ftol": 1e-6},
    )
    best_theta = res.x if np.all(np.isfinite(res.x)) else init
    sol_full = solve_theta(best_theta, t_eval_full)
    if sol_full is None:
        sol_full = solve_theta(init, t_eval_full)
        if sol_full is None:
            return None

    pred_cases = sol_full.y[5, :]
    pred_deaths = sol_full.y[4, :]
    start_idx = int((cutoff - artifact.data_start_date).days)
    if start_idx < 0 or start_idx >= len(pred_cases):
        return None

    intervals: Dict[int, Tuple[float, float]] = {}
    for h in range(1, months + 1):
        end_h = cutoff + relativedelta(months=h)
        end_idx = int((end_h - artifact.data_start_date).days)
        if end_idx < 0 or end_idx >= len(pred_cases):
            return None
        intervals[h] = (
            float(max(pred_cases[end_idx] - pred_cases[start_idx], 0.0)),
            float(max(pred_deaths[end_idx] - pred_deaths[start_idx], 0.0)),
        )
    return intervals


def _robust_statistic(values: Sequence[float], mode: str, trim_quantile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if mode == "median":
        return float(np.median(arr))
    if mode == "trimmed_mean":
        q = float(min(max(trim_quantile, 0.0), 0.45))
        lo = np.quantile(arr, q)
        hi = np.quantile(arr, 1.0 - q)
        arr_trim = arr[(arr >= lo) & (arr <= hi)]
        if arr_trim.size == 0:
            arr_trim = arr
        return float(np.mean(arr_trim))
    return float(np.mean(arr))


def _estimate_train_gammas(
    train_region_ids: List[str],
    artifacts: Dict[str, RegionArtifact],
    args: argparse.Namespace,
    horizon: int,
) -> Dict[str, float]:
    default_gammas = dict(sorted(default_dict_normalized_policy_gamma.items(), key=lambda x: x[0]))
    z_region_values: List[float] = []
    gamma_sum_total = {p: 0.0 for p in default_gammas}
    gamma_count_total = {p: 0 for p in default_gammas}
    per_policy_region_means = {p: [] for p in default_gammas}

    for rid in train_region_ids:
        art = artifacts[rid]
        z_count_h = int(art.z_count_by_horizon.get(horizon, 0))
        z_sum_h = float(art.z_sum_by_horizon.get(horizon, 0.0))
        if z_count_h > 0:
            z_region_values.append(float(z_sum_h / z_count_h))
        gamma_sum_h = art.gamma_sum_by_horizon.get(horizon, {})
        gamma_count_h = art.gamma_count_by_horizon.get(horizon, {})
        for p in default_gammas:
            c = int(gamma_count_h.get(p, 0))
            if c > 0:
                gamma_sum_total[p] += float(gamma_sum_h[p])
                gamma_count_total[p] += c
                per_policy_region_means[p].append(float(gamma_sum_h[p] / c))

    if len(z_region_values) == 0:
        z_center = 1.0
    elif args.z_stat == "median":
        z_center = float(np.median(z_region_values))
    else:
        z_center = float(np.mean(z_region_values))

    out: Dict[str, float] = {}
    if args.gamma_estimator == "paper_day_mean":
        # Exact manuscript estimator:
        # gamma_i = sum_R sum_{t in D_{R,i}} gamma_R(t) / sum_R |D_{R,i}|
        for p, default_gamma in default_gammas.items():
            c = int(gamma_count_total[p])
            if c > 0:
                gamma_hat = float(gamma_sum_total[p] / c)
            else:
                # Backoff only when policy i is unseen in train split for this horizon.
                gamma_hat = float(default_gamma * z_center)
            out[p] = float(np.clip(gamma_hat, args.gamma_floor, args.gamma_cap))
    else:
        for p, default_gamma in default_gammas.items():
            means = per_policy_region_means[p]
            n_regions = len(means)
            prior = float(default_gamma * z_center)
            if n_regions == 0:
                gamma_hat = prior
            else:
                center = _robust_statistic(means, args.robust_stat, args.trim_quantile)
                if n_regions < args.min_policy_regions:
                    w = n_regions / (n_regions + args.shrinkage_regions + args.min_policy_regions)
                else:
                    w = n_regions / (n_regions + args.shrinkage_regions)
                gamma_hat = (1.0 - w) * prior + w * center
            out[p] = float(np.clip(gamma_hat, args.gamma_floor, args.gamma_cap))
    return out


def _estimate_region_k_from_precutoff(
    base_gammas: Dict[str, float],
    artifact: RegionArtifact,
    args: argparse.Namespace,
) -> Optional[float]:
    if args.k_estimation_mode == "cutoff_policy":
        ref_policy = artifact.policy_at_cutoff
        if ref_policy not in base_gammas:
            ref_policy = artifact.policy_vector[0] if len(artifact.policy_vector) > 0 else None
        if ref_policy is None or ref_policy not in base_gammas:
            return None
        ref_train_gamma = float(base_gammas[ref_policy])
        ref_region_gamma = float(_gamma_t_for_day(artifact.cutoff_date, artifact))
        if (not np.isfinite(ref_train_gamma)) or (ref_train_gamma <= 0):
            return None
        if (not np.isfinite(ref_region_gamma)) or (ref_region_gamma <= 0):
            return None
        k_hat = float(ref_region_gamma / ref_train_gamma)
        s = float(args.k_shrinkage_to_one)
        k_hat = (1.0 - s) * k_hat + s * 1.0
        k_hat = float(np.clip(k_hat, args.k_floor, args.k_cap))
        return k_hat

    # Fit k_R over all pre-cutoff days where policy labels are observed:
    # gamma_{R,i} ~= k_R * gamma_i (weighted by pre-cutoff day counts).
    num = 0.0
    den = 0.0
    for p, gamma_i in base_gammas.items():
        c = int(artifact.gamma_count_precutoff.get(p, 0))
        if c <= 0:
            continue
        s = float(artifact.gamma_sum_precutoff.get(p, 0.0))
        gamma_i = float(gamma_i)
        num += gamma_i * s
        den += c * (gamma_i ** 2)
    if den <= 0:
        return None
    k_hat = float(num / den)
    if (not np.isfinite(k_hat)) or (k_hat <= 0):
        return None
    s = float(args.k_shrinkage_to_one)
    k_hat = (1.0 - s) * k_hat + s * 1.0
    k_hat = float(np.clip(k_hat, args.k_floor, args.k_cap))
    return k_hat


def _transfer_gammas_for_region(
    base_gammas: Dict[str, float],
    artifact: RegionArtifact,
    args: argparse.Namespace,
) -> Dict[str, float]:
    if args.transfer_gamma_mode == "absolute":
        return base_gammas

    k_hat = _estimate_region_k_from_precutoff(base_gammas, artifact, args)
    if k_hat is None:
        return base_gammas

    floor = float(args.relative_gamma_floor)
    cap = float(args.relative_gamma_cap)
    out: Dict[str, float] = {}
    for p, g in base_gammas.items():
        gamma_val = float(k_hat * float(g))
        gamma_val = max(gamma_val, floor)
        gamma_val = min(gamma_val, cap)
        out[p] = float(gamma_val)
    return out


def _ci95_mean(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    center = float(np.mean(arr))
    if arr.size == 1:
        return center, center, center
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    half = float(stats.t.ppf(0.975, df=arr.size - 1) * se)
    return center, center - half, center + half


def _prepare_artifacts_for_cutoff(
    cfg: CutoffConfig,
    truth_df: pd.DataFrame,
    pop_lookup: Dict[Tuple[str, str], float],
    max_months: int,
    min_precutoff_days: int,
    max_regions: int,
) -> Dict[str, RegionArtifact]:
    cutoff_dt = _to_date(cfg.cutoff_date)
    end_dt = cutoff_dt + relativedelta(months=max_months)
    params = pd.read_csv(cfg.param_file, keep_default_na=False)
    params["Country"] = params["Country"].astype(str)
    params["Province"] = params["Province"].fillna("None").astype(str)
    params["Data Start Date"] = pd.to_datetime(params["Data Start Date"], errors="coerce").dt.normalize()
    params = params.dropna(subset=["Data Start Date"]).sort_values(["Country", "Province", "Data Start Date"])
    params = params.drop_duplicates(subset=["Country", "Province"], keep="last").reset_index(drop=True)

    truth_group = {(c, p): g.sort_values("date").reset_index(drop=True) for (c, p), g in truth_df.groupby(["country", "province"])}

    artifacts: Dict[str, RegionArtifact] = {}
    for _, row in params.iterrows():
        country = str(row["Country"])
        province = str(row["Province"])
        continent = str(row["Continent"])
        key = (country, province)
        if key not in pop_lookup:
            continue
        if key not in truth_group:
            continue
        region_truth = truth_group[key]
        data_start = row["Data Start Date"].to_pydatetime()
        data_start = pd.to_datetime(data_start).normalize().to_pydatetime()
        idx = region_truth.set_index("date")
        required_dates = [pd.to_datetime(data_start), pd.to_datetime(cutoff_dt)] + [
            pd.to_datetime(cutoff_dt + relativedelta(months=h)) for h in range(1, max_months + 1)
        ]
        if any(d not in idx.index for d in required_dates):
            continue

        start_policy = max(data_start, datetime(2020, 3, 1))
        policy_df = _read_policy_data_cached(
            country=country,
            province=province,
            start_date=start_policy.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
        )
        if policy_df is None or policy_df.empty:
            continue

        policy_pre = policy_df[policy_df["date"] <= pd.to_datetime(cutoff_dt)].copy()
        if policy_pre["date"].nunique() < min_precutoff_days:
            continue

        policy_cutoff = policy_df[policy_df["date"] == pd.to_datetime(cutoff_dt)].copy()
        if policy_cutoff.empty:
            policy_cutoff = policy_df[policy_df["date"] >= pd.to_datetime(cutoff_dt)].head(1).copy()
        if policy_cutoff.empty:
            continue
        policy_at_cutoff = _active_policy_from_row(policy_cutoff.iloc[0])
        if policy_at_cutoff is None:
            continue

        policy_vector: List[str] = []
        valid_vector = True
        for m in range(max_months):
            m_start = cutoff_dt + relativedelta(months=m)
            m_end = cutoff_dt + relativedelta(months=m + 1) - timedelta(days=1)
            dom = _dominant_policy(policy_df, m_start, m_end)
            if dom is None:
                valid_vector = False
                break
            policy_vector.append(dom)
        if not valid_vector:
            continue

        actual_cases_by_horizon: Dict[int, float] = {}
        actual_deaths_by_horizon: Dict[int, float] = {}
        start_cases = float(idx.loc[pd.to_datetime(cutoff_dt), "case_cnt"])
        start_deaths = float(idx.loc[pd.to_datetime(cutoff_dt), "death_cnt"])
        for h in range(1, max_months + 1):
            e_dt = pd.to_datetime(cutoff_dt + relativedelta(months=h))
            actual_cases_by_horizon[h] = float(idx.loc[e_dt, "case_cnt"] - start_cases)
            actual_deaths_by_horizon[h] = float(idx.loc[e_dt, "death_cnt"] - start_deaths)

        artifact = RegionArtifact(
            region_id=_region_id(country, province),
            continent=continent,
            country=country,
            province=province,
            cutoff_label=cfg.label,
            cutoff_date=cutoff_dt,
            data_start_date=data_start,
            model_version=cfg.model_version,
            delphi_params=_parse_delphi_params(row, cfg.model_version),
            population=float(pop_lookup[key]),
            initial_cases=float(idx.loc[pd.to_datetime(data_start), "case_cnt"]),
            initial_deaths=float(idx.loc[pd.to_datetime(data_start), "death_cnt"]),
            policy_at_cutoff=policy_at_cutoff,
            policy_vector=policy_vector,
            actual_cases_by_horizon=actual_cases_by_horizon,
            actual_deaths_by_horizon=actual_deaths_by_horizon,
            gamma_sum_precutoff={},
            gamma_count_precutoff={},
            gamma_sum_by_horizon={},
            gamma_count_by_horizon={},
            z_sum_by_horizon={},
            z_count_by_horizon={},
            region_truth=region_truth,
            baseline_predictions={},
        )

        policy_pre_k = policy_df[policy_df["date"] < pd.to_datetime(cutoff_dt)].copy()
        if policy_pre_k.empty:
            policy_pre_k = policy_pre.copy()
        gamma_sum_pre, gamma_count_pre, _, z_count_pre = _build_gamma_stats(artifact, policy_pre_k)
        if z_count_pre <= 0:
            continue
        artifact.gamma_sum_precutoff = gamma_sum_pre
        artifact.gamma_count_precutoff = gamma_count_pre

        # Gamma estimation stats are computed on the evaluation window (D -> D+X)
        # because this benchmark targets transfer across regions, not transfer across time.
        valid_gamma_stats = True
        for h in range(1, max_months + 1):
            eval_end = cutoff_dt + relativedelta(months=h)
            policy_eval = policy_df[
                (policy_df["date"] >= pd.to_datetime(cutoff_dt))
                & (policy_df["date"] <= pd.to_datetime(eval_end))
            ].copy()
            if policy_eval.empty:
                valid_gamma_stats = False
                break
            gamma_sum, gamma_count, z_sum, z_count = _build_gamma_stats(artifact, policy_eval)
            if z_count <= 0:
                valid_gamma_stats = False
                break
            artifact.gamma_sum_by_horizon[h] = gamma_sum
            artifact.gamma_count_by_horizon[h] = gamma_count
            artifact.z_sum_by_horizon[h] = float(z_sum)
            artifact.z_count_by_horizon[h] = int(z_count)

        if not valid_gamma_stats:
            continue

        artifacts[artifact.region_id] = artifact

    if max_regions > 0 and len(artifacts) > max_regions:
        selected = sorted(artifacts.keys())[:max_regions]
        artifacts = {rid: artifacts[rid] for rid in selected}
    return artifacts


def _precompute_baselines(artifacts: Dict[str, RegionArtifact], max_months: int, min_seir_train_days: int) -> Dict[str, RegionArtifact]:
    keep: Dict[str, RegionArtifact] = {}
    for rid, art in artifacts.items():
        pred_full = _simulate_delphi(art, months=max_months, mode="full")
        pred_const = _simulate_delphi(art, months=max_months, mode="constant")
        pred_seir = _fit_predict_seir(art, months=max_months, min_train_days=min_seir_train_days)
        if pred_full is None or pred_const is None or pred_seir is None:
            continue
        art.baseline_predictions = {
            "delphi_full": pred_full,
            "delphi_constant": pred_const,
            "seir": pred_seir,
        }
        keep[rid] = art
    return keep


def _build_split_metrics(detail_df: pd.DataFrame) -> pd.DataFrame:
    return (
        detail_df.groupby(["cutoff_label", "horizon_months", "model", "split_id"], as_index=False)
        .agg(
            n_regions_cases=("ape_cases_pct", "count"),
            n_regions_deaths=("ape_deaths_pct", "count"),
            mean_ape_cases_pct=("ape_cases_pct", "mean"),
            median_ape_cases_pct=("ape_cases_pct", "median"),
            mean_ape_deaths_pct=("ape_deaths_pct", "mean"),
            median_ape_deaths_pct=("ape_deaths_pct", "median"),
        )
        .sort_values(["cutoff_label", "horizon_months", "model", "split_id"])
    )


def _summarize_with_ci(split_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    group_cols = ["cutoff_label", "horizon_months", "model"]
    for (cutoff, horizon, model), g in split_metrics.groupby(group_cols):
        for metric_kind in ["mean", "median"]:
            for outcome in ["cases", "deaths"]:
                col = f"{metric_kind}_ape_{outcome}_pct"
                center, ci_low, ci_high = _ci95_mean(g[col].to_numpy(dtype=float))
                rows.append(
                    {
                        "cutoff_label": cutoff,
                        "horizon_months": int(horizon),
                        "model": model,
                        "metric_kind": metric_kind,
                        "outcome": outcome,
                        "center_ape_pct": center,
                        "ci95_low_ape_pct": ci_low,
                        "ci95_high_ape_pct": ci_high,
                        "n_splits": int(g["split_id"].nunique()),
                    }
                )
    return pd.DataFrame(rows).sort_values(["metric_kind", "outcome", "horizon_months", "cutoff_label", "model"])


def _paired_significance_x1(split_metrics: pd.DataFrame, metric_kind: str) -> pd.DataFrame:
    rows: List[dict] = []
    base_subset = split_metrics[split_metrics["horizon_months"] == 1].copy()
    comparisons = ["delphi_constant", "seir"]
    for cutoff in sorted(base_subset["cutoff_label"].unique().tolist()):
        cutoff_df = base_subset[base_subset["cutoff_label"] == cutoff]
        for baseline in comparisons:
            for outcome in ["cases", "deaths"]:
                col = f"{metric_kind}_ape_{outcome}_pct"
                left = cutoff_df[cutoff_df["model"] == "themis_transfer"][["split_id", col]].rename(columns={col: "themis"})
                right = cutoff_df[cutoff_df["model"] == baseline][["split_id", col]].rename(columns={col: "baseline"})
                pair = left.merge(right, on="split_id", how="inner")
                if pair.empty:
                    continue
                diff = pair["themis"].to_numpy(dtype=float) - pair["baseline"].to_numpy(dtype=float)
                diff = diff[np.isfinite(diff)]
                if diff.size == 0:
                    continue
                mean_diff, ci_low, ci_high = _ci95_mean(diff)
                p_two = np.nan
                p_one = np.nan
                if diff.size >= 2:
                    test_res = stats.ttest_rel(pair["themis"], pair["baseline"], nan_policy="omit")
                    if np.isfinite(test_res.pvalue):
                        p_two = float(test_res.pvalue)
                        p_one = float(p_two / 2.0) if mean_diff < 0 else float(1.0 - p_two / 2.0)
                rows.append(
                    {
                        "cutoff_label": cutoff,
                        "horizon_months": 1,
                        "metric_kind": metric_kind,
                        "outcome": outcome,
                        "compare_themis_vs": baseline,
                        "n_splits": int(diff.size),
                        "mean_diff_themis_minus_baseline_pct": mean_diff,
                        "ci95_low_diff_pct": ci_low,
                        "ci95_high_diff_pct": ci_high,
                        "pvalue_two_sided": p_two,
                        "pvalue_one_sided_themis_better": p_one,
                        "significant_better_95pct": bool((ci_high < 0) and np.isfinite(p_one) and (p_one < 0.05)),
                    }
                )

    for baseline in comparisons:
        pooled = base_subset[base_subset["model"].isin(["themis_transfer", baseline])].copy()
        for outcome in ["cases", "deaths"]:
            col = f"{metric_kind}_ape_{outcome}_pct"
            left = pooled[pooled["model"] == "themis_transfer"][["cutoff_label", "split_id", col]].rename(columns={col: "themis"})
            right = pooled[pooled["model"] == baseline][["cutoff_label", "split_id", col]].rename(columns={col: "baseline"})
            pair = left.merge(right, on=["cutoff_label", "split_id"], how="inner")
            if pair.empty:
                continue
            diff = pair["themis"].to_numpy(dtype=float) - pair["baseline"].to_numpy(dtype=float)
            diff = diff[np.isfinite(diff)]
            if diff.size == 0:
                continue
            mean_diff, ci_low, ci_high = _ci95_mean(diff)
            p_two = np.nan
            p_one = np.nan
            if diff.size >= 2:
                test_res = stats.ttest_rel(pair["themis"], pair["baseline"], nan_policy="omit")
                if np.isfinite(test_res.pvalue):
                    p_two = float(test_res.pvalue)
                    p_one = float(p_two / 2.0) if mean_diff < 0 else float(1.0 - p_two / 2.0)
            rows.append(
                {
                    "cutoff_label": "pooled_all_dates",
                    "horizon_months": 1,
                    "metric_kind": metric_kind,
                    "outcome": outcome,
                    "compare_themis_vs": baseline,
                    "n_splits": int(diff.size),
                    "mean_diff_themis_minus_baseline_pct": mean_diff,
                    "ci95_low_diff_pct": ci_low,
                    "ci95_high_diff_pct": ci_high,
                    "pvalue_two_sided": p_two,
                    "pvalue_one_sided_themis_better": p_one,
                    "significant_better_95pct": bool((ci_high < 0) and np.isfinite(p_one) and (p_one < 0.05)),
                }
            )

    if len(rows) == 0:
        return pd.DataFrame(
            columns=[
                "cutoff_label",
                "horizon_months",
                "metric_kind",
                "outcome",
                "compare_themis_vs",
                "n_splits",
                "mean_diff_themis_minus_baseline_pct",
                "ci95_low_diff_pct",
                "ci95_high_diff_pct",
                "pvalue_two_sided",
                "pvalue_one_sided_themis_better",
                "significant_better_95pct",
            ]
        )
    return pd.DataFrame(rows)


def _plot_metric_grid(
    summary_df: pd.DataFrame,
    cutoff_order: List[str],
    metric_kind: str,
    outcome: str,
    max_months: int,
    out_path: Path,
) -> None:
    subset = summary_df[(summary_df["metric_kind"] == metric_kind) & (summary_df["outcome"] == outcome)].copy()
    if subset.empty:
        return
    horizons = list(range(1, max_months + 1))
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4.6), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    width = 0.18
    x = np.arange(len(cutoff_order), dtype=float)
    for ax, h in zip(axes, horizons):
        hdf = subset[subset["horizon_months"] == h]
        for j, model in enumerate(MODEL_ORDER):
            vals: List[float] = []
            err_low: List[float] = []
            err_high: List[float] = []
            for cutoff in cutoff_order:
                row = hdf[(hdf["cutoff_label"] == cutoff) & (hdf["model"] == model)]
                if row.empty:
                    vals.append(np.nan)
                    err_low.append(np.nan)
                    err_high.append(np.nan)
                    continue
                center = float(row.iloc[0]["center_ape_pct"])
                low = float(row.iloc[0]["ci95_low_ape_pct"])
                high = float(row.iloc[0]["ci95_high_ape_pct"])
                vals.append(center)
                err_low.append(max(center - low, 0.0) if np.isfinite(low) else np.nan)
                err_high.append(max(high - center, 0.0) if np.isfinite(high) else np.nan)

            positions = x + (j - (len(MODEL_ORDER) - 1) / 2.0) * width
            ax.bar(positions, vals, width=width, color=MODEL_COLORS[model], alpha=0.88, label=MODEL_LABELS[model])
            ax.errorbar(
                positions,
                vals,
                yerr=[err_low, err_high],
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=2.5,
            )

        ax.set_title(f"Horizon X={h}")
        ax.set_xticks(x)
        ax.set_xticklabels([c[-4:] for c in cutoff_order])
        ax.set_xlabel("Cutoff (MMDD)")
        ax.grid(alpha=0.25, axis="y")

    axes[0].set_ylabel(f"{metric_kind.title()} APE (%) - {outcome.title()}")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.train_fraction <= 0 or args.train_fraction > MAX_TRAIN_FRACTION:
        raise ValueError(f"train_fraction must be in (0, {MAX_TRAIN_FRACTION}].")
    if args.max_months < 1:
        raise ValueError("max_months must be >= 1.")
    if args.min_eval_actual_cases < 0 or args.min_eval_actual_deaths < 0:
        raise ValueError("min-eval-actual-cases/deaths must be >= 0.")
    if args.relative_gamma_floor < 0:
        raise ValueError("relative-gamma-floor must be >= 0.")
    if args.relative_gamma_cap <= 0 or args.relative_gamma_cap < args.relative_gamma_floor:
        raise ValueError("relative-gamma-cap must be >= relative-gamma-floor and > 0.")
    if args.k_floor < 0:
        raise ValueError("k-floor must be >= 0.")
    if args.k_cap <= 0 or args.k_cap < args.k_floor:
        raise ValueError("k-cap must be >= k-floor and > 0.")
    if args.k_shrinkage_to_one < 0 or args.k_shrinkage_to_one > 1:
        raise ValueError("k-shrinkage-to-one must be in [0,1].")

    selected_cutoffs = [c.strip() for c in args.cutoffs.split(",") if c.strip()]
    for c in selected_cutoffs:
        if c not in CUTOFF_CONFIGS:
            raise ValueError(f"Unknown cutoff {c}. Allowed: {sorted(CUTOFF_CONFIGS.keys())}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"themis_transfer_multidate_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    truth_df = _load_truth(args.truth_path)
    pop_lookup = _build_population_lookup()
    rng = np.random.default_rng(args.seed)

    artifacts_by_cutoff: Dict[str, Dict[str, RegionArtifact]] = {}
    cutoff_region_counts: Dict[str, int] = {}

    print("Preparing cutoff-specific region artifacts...")
    for label in selected_cutoffs:
        cfg = CUTOFF_CONFIGS[label]
        artifacts = _prepare_artifacts_for_cutoff(
            cfg=cfg,
            truth_df=truth_df,
            pop_lookup=pop_lookup,
            max_months=args.max_months,
            min_precutoff_days=args.min_precutoff_days,
            max_regions=args.max_regions,
        )
        artifacts = _precompute_baselines(
            artifacts=artifacts,
            max_months=args.max_months,
            min_seir_train_days=args.min_seir_train_days,
        )
        if len(artifacts) < 8:
            print(f"[WARN] Cutoff {label}: only {len(artifacts)} usable regions after filtering; skipping cutoff.")
            continue
        artifacts_by_cutoff[label] = artifacts
        cutoff_region_counts[label] = len(artifacts)
        print(f"Cutoff {label}: {len(artifacts)} usable regions.")

    if len(artifacts_by_cutoff) == 0:
        raise RuntimeError("No cutoff retained enough regions for benchmarking.")

    detail_rows: List[dict] = []
    for label, artifacts in artifacts_by_cutoff.items():
        region_ids = sorted(artifacts.keys())
        n_regions = len(region_ids)
        test_size = max(1, int(round(n_regions * (1.0 - args.train_fraction))))
        test_size = min(test_size, n_regions - 1)
        print(f"Running splits for cutoff {label}: regions={n_regions}, test_size={test_size}, splits={args.n_splits}")

        for split_id in range(1, args.n_splits + 1):
            test_regions = sorted(rng.choice(region_ids, size=test_size, replace=False).tolist())
            test_set = set(test_regions)
            train_regions = [rid for rid in region_ids if rid not in test_set]

            themis_gammas_by_horizon = {
                h: _estimate_train_gammas(train_regions, artifacts, args, horizon=h)
                for h in range(1, args.max_months + 1)
            }
            for rid in test_regions:
                art = artifacts[rid]
                themis_pred_by_horizon: Dict[int, Tuple[float, float]] = {}
                valid_themis = True
                for h in range(1, args.max_months + 1):
                    region_gammas_h = _transfer_gammas_for_region(themis_gammas_by_horizon[h], art, args)
                    themis_pred_h = _simulate_delphi(
                        art,
                        months=h,
                        mode="themis_transfer",
                        themis_gammas=region_gammas_h,
                    )
                    if themis_pred_h is None or h not in themis_pred_h:
                        valid_themis = False
                        break
                    themis_pred_by_horizon[h] = themis_pred_h[h]
                if not valid_themis:
                    continue

                for h in range(1, args.max_months + 1):
                    model_to_pred = {
                        "themis_transfer": themis_pred_by_horizon[h],
                        "delphi_full": art.baseline_predictions["delphi_full"][h],
                        "delphi_constant": art.baseline_predictions["delphi_constant"][h],
                        "seir": art.baseline_predictions["seir"][h],
                    }
                    for model_name, pred_pair in model_to_pred.items():
                        pred_cases, pred_deaths = pred_pair
                        actual_cases = art.actual_cases_by_horizon[h]
                        actual_deaths = art.actual_deaths_by_horizon[h]
                        ape_cases = (
                            _safe_ape(pred_cases, actual_cases)
                            if actual_cases >= args.min_eval_actual_cases
                            else np.nan
                        )
                        ape_deaths = (
                            _safe_ape(pred_deaths, actual_deaths)
                            if actual_deaths >= args.min_eval_actual_deaths
                            else np.nan
                        )
                        detail_rows.append(
                            {
                                "cutoff_label": label,
                                "cutoff_date": art.cutoff_date.strftime("%Y-%m-%d"),
                                "model_version": art.model_version,
                                "split_id": split_id,
                                "horizon_months": h,
                                "model": model_name,
                                "test_region_id": rid,
                                "continent": art.continent,
                                "country": art.country,
                                "province": art.province,
                                "pred_cases": pred_cases,
                                "actual_cases": actual_cases,
                                "ape_cases_pct": ape_cases,
                                "pred_deaths": pred_deaths,
                                "actual_deaths": actual_deaths,
                                "ape_deaths_pct": ape_deaths,
                            }
                        )
            if split_id % 5 == 0:
                print(f"  cutoff={label} completed split {split_id}/{args.n_splits}")

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        raise RuntimeError("No benchmark rows were generated.")

    split_metrics = _build_split_metrics(detail_df)
    summary_df = _summarize_with_ci(split_metrics)
    significance_df = _paired_significance_x1(split_metrics, metric_kind=args.primary_metric)

    detail_df.to_csv(run_dir / "benchmark_detail.csv", index=False)
    split_metrics.to_csv(run_dir / "benchmark_split_metrics.csv", index=False)
    summary_df.to_csv(run_dir / "benchmark_summary_with_ci.csv", index=False)
    significance_df.to_csv(run_dir / "benchmark_significance_x1.csv", index=False)

    cutoff_order = [c for c in selected_cutoffs if c in artifacts_by_cutoff]
    _plot_metric_grid(
        summary_df=summary_df,
        cutoff_order=cutoff_order,
        metric_kind=args.primary_metric,
        outcome="cases",
        max_months=args.max_months,
        out_path=run_dir / f"plot_{args.primary_metric}_ape_cases.png",
    )
    _plot_metric_grid(
        summary_df=summary_df,
        cutoff_order=cutoff_order,
        metric_kind=args.primary_metric,
        outcome="deaths",
        max_months=args.max_months,
        out_path=run_dir / f"plot_{args.primary_metric}_ape_deaths.png",
    )

    config_out = {
        "cutoffs_requested": selected_cutoffs,
        "cutoffs_used": cutoff_order,
        "cutoff_region_counts": cutoff_region_counts,
        "n_splits": args.n_splits,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "max_months": args.max_months,
        "max_regions": args.max_regions,
        "primary_metric": args.primary_metric,
        "gamma_estimation_window": "evaluation_period_by_horizon_D_to_DplusX",
        "gamma_estimator": args.gamma_estimator,
        "transfer_gamma_mode": args.transfer_gamma_mode,
        "k_estimation_mode": args.k_estimation_mode,
        "k_floor": args.k_floor,
        "k_cap": args.k_cap,
        "k_shrinkage_to_one": args.k_shrinkage_to_one,
        "relative_gamma_floor": args.relative_gamma_floor,
        "relative_gamma_cap": args.relative_gamma_cap,
        "min_eval_actual_cases": args.min_eval_actual_cases,
        "min_eval_actual_deaths": args.min_eval_actual_deaths,
        "robust_gamma_config": {
            "robust_stat": args.robust_stat,
            "trim_quantile": args.trim_quantile,
            "z_stat": args.z_stat,
            "min_policy_regions": args.min_policy_regions,
            "shrinkage_regions": args.shrinkage_regions,
            "gamma_floor": args.gamma_floor,
            "gamma_cap": args.gamma_cap,
        },
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    pooled_sig = significance_df[
        (significance_df["cutoff_label"] == "pooled_all_dates")
        & (significance_df["compare_themis_vs"].isin(["delphi_constant", "seir"]))
    ]
    success_x1 = False
    if not pooled_sig.empty:
        success_x1 = bool(pooled_sig["significant_better_95pct"].all())

    print(f"Output directory: {run_dir}")
    print(f"Rows in detail table: {len(detail_df)}")
    print(f"Rows in split-metric table: {len(split_metrics)}")
    print(
        f"Pooled X=1 significance vs DELPHI-constant and SEIR "
        f"({args.primary_metric} APE, both outcomes):",
        success_x1,
    )
    if not pooled_sig.empty:
        print(pooled_sig[["metric_kind", "outcome", "compare_themis_vs", "mean_diff_themis_minus_baseline_pct", "ci95_low_diff_pct", "ci95_high_diff_pct", "pvalue_one_sided_themis_better", "significant_better_95pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
