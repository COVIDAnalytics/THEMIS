import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from analyze_gamma_rank import build_gamma_matrix as _build_global_gamma_matrix
from analyze_gamma_rank import rank1_imputation as _als_rank1_imputation
from pandemic_functions.delphi_functions.DELPHI_model import model_covid
from pandemic_functions.delphi_functions.DELPHI_model_fitting import solve_and_predict_area
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    past_parameters as _past_parameters_df,
    read_oxford_country_policy_data,
    read_policy_data_us_only,
)
from pandemic_functions.delphi_functions.DELPHI_utils import gamma_t, get_initial_conditions
from pandemic_functions.pandemic_params import (
    default_dict_normalized_policy_gamma,
    future_policies,
    global_populations,
    p_d,
    p_h,
    p_v,
    region_symbol_country_dict,
    region_symbol_continent_dict,
)


MODEL_ORDER = ["delphi_lookahead", "themis", "delphi", "delphi_constant", "sir", "seir", "seird"]


@dataclass
class RegionContext:
    region: str
    country: str
    province: str
    population: float
    cases: pd.DataFrame


@dataclass
class FitOutput:
    params_row: pd.Series
    preds_df: pd.DataFrame
    params_tuple: tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time-transfer counterfactual benchmark on the same region "
            "(train: 2020-03-15..2020-06-15, eval: 2020-06-15..2020-09-15)."
        )
    )
    parser.add_argument("--train-start", type=str, default="2020-03-15")
    parser.add_argument("--train-end", type=str, default="2020-06-15")
    parser.add_argument("--eval-start", type=str, default=None,
                        help="Override eval start date (default: same as train-end).")
    parser.add_argument("--eval-end", type=str, default="2020-09-15")
    parser.add_argument(
        "--gamma-policy-days-thresh",
        type=int,
        default=20,
        help="Policy minimum day-count threshold for direct region gamma estimates.",
    )
    parser.add_argument(
        "--seir-min-train-days",
        type=int,
        default=21,
        help="Minimum train days to fit SEIR.",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="",
        help="Comma-separated region IDs like 'France' or 'US - Florida' (default: all).",
    )
    parser.add_argument(
        "--optimization-method",
        type=str,
        default="annealing",
        choices=["annealing", "tnc"],
        help="DELPHI optimization method (annealing is more robust but slower).",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=100,
        help="Minimum cumulative cases in window to include a region.",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--ci-alpha",
        type=float,
        default=0.05,
        help="Significance level for confidence intervals (default 0.05 = 95%% CI).",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=0,
        help="If >0, randomly sample this many regions from the filtered pool (deterministic via --random-seed).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for --random-sample so the selection is reproducible.",
    )
    parser.add_argument(
        "--fit-quality-gate",
        action="store_true",
        default=True,
        help="Skip regions whose DELPHI training fit fails quality checks (enabled by default).",
    )
    parser.add_argument(
        "--no-fit-quality-gate",
        dest="fit_quality_gate",
        action="store_false",
        help="Disable the DELPHI-fit quality gate (evaluate every region that reaches the fit step).",
    )
    parser.add_argument(
        "--fit-quality-max-train-mape",
        type=float,
        default=40.0,
        help="Training MAPE (last-15-days avg) above which a region is rejected for THEMIS.",
    )
    parser.add_argument(
        "--fit-quality-max-corners",
        type=int,
        default=1,
        help="Max number of parameter corners hit (alpha@UB, r_s@LB, |DoA|@bound) before rejecting the fit.",
    )
    parser.add_argument(
        "--themis-gamma-cap-ratio",
        type=float,
        default=2.5,
        help=(
            "Cap THEMIS piecewise gamma at cap_ratio * arctan_gamma(t). "
            "0 or negative disables the cap; larger values allow more policy-gamma freedom."
        ),
    )
    parser.add_argument(
        "--use-past-params",
        action="store_true",
        default=False,
        help=(
            "Use pre-fitted past_parameters for DELPHI instead of fresh fitting. "
            "Runs the ODE from each region's Data Start Date to ensure params are "
            "on the same scale as the rank-1 imputed gammas."
        ),
    )
    parser.add_argument(
        "--fresh-fit-cache",
        type=str,
        default="",
        help=(
            "Path to directory produced by parallel_fit_delphi.py. "
            "Loads pre-computed fresh params + rank-1 gamma matrix from cache "
            "instead of using past_parameters or fitting on the fly."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="simulation_results")
    return parser.parse_args()


def _to_date(x: str) -> datetime:
    return pd.to_datetime(x).normalize().to_pydatetime()


CASE_DATA_DIR = Path("pandemic_functions/pandemic_data")

# Oxford policy data uses different country names than the case file names.
# read_oxford_country_policy_data already handles US->United States etc.,
# but the case files use JHU names. This maps JHU case-file country names to
# the name that read_oxford_country_policy_data expects as input.
_CASE_TO_POLICY_COUNTRY = {
    "Korea, South": "Korea, South",
    "Congo (Kinshasa)": "Congo (Kinshasa)",
    "Congo (Brazzaville)": "Congo (Brazzaville)",
    "Czechia": "Czechia",
    "Slovakia": "Slovakia",
    "Cote d'Ivoire": "Cote d'Ivoire",
    "Eswatini": "Eswatini",
    "Burma": "Myanmar",
    "Taiwan": "Taiwan",
    "West Bank and Gaza": "West Bank and Gaza",
}

# Regions that should be skipped (not real countries/states)
_SKIP_REGIONS = {
    "Antarctica", "Diamond Princess", "MS Zaandam",
    "Summer Olympics 2020", "Winter Olympics 2022",
    "Western Sahara", "Holy See",
}


def _safe_ape(pred: float, actual: float) -> float:
    if not np.isfinite(actual) or actual <= 0:
        return np.nan
    return abs(pred - actual) / abs(actual) * 100.0


def _parse_case_file(path: Path) -> Optional[Tuple[str, str]]:
    """Read the first row of a Cases CSV to extract (country, province)."""
    try:
        row = pd.read_csv(path, nrows=1, keep_default_na=False)
        country = str(row["country"].iloc[0]).strip()
        province = str(row["province"].iloc[0]).strip()
        if not province:
            province = "None"
        return (country, province)
    except Exception:
        return None


def _discover_all_regions() -> List[Tuple[str, str, Path]]:
    """Scan case data directory for all Cases_*.csv files and return (country, province, path)."""
    results = []
    for p in sorted(CASE_DATA_DIR.glob("Cases_*.csv")):
        if p.name == "Cases_Recovered.csv":
            continue
        parsed = _parse_case_file(p)
        if parsed is None:
            continue
        country, province = parsed
        if country in _SKIP_REGIONS or province in _SKIP_REGIONS:
            continue
        results.append((country, province, p))
    return results


def _build_population_lookup() -> Dict[Tuple[str, str], float]:
    pop = global_populations.copy()
    pop["Country"] = pop["Country"].astype(str)
    pop["Province"] = pop["Province"].fillna("None").astype(str)
    pop["pop2016"] = pd.to_numeric(pop["pop2016"], errors="coerce")
    pop = pop.dropna(subset=["pop2016"]).copy()
    pop = pop.sort_values(["Country", "Province", "pop2016"]).drop_duplicates(
        subset=["Country", "Province"], keep="last"
    )
    return {(str(r.Country), str(r.Province)): float(r.pop2016) for _, r in pop.iterrows()}


def _region_id(country: str, province: str) -> str:
    if province == "None":
        return country
    return f"{country} - {province}"


def _load_cases(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["case_cnt"] = pd.to_numeric(df["case_cnt"], errors="coerce")
    df["death_cnt"] = pd.to_numeric(df["death_cnt"], errors="coerce")
    for col in ["total_hospitalization", "people_vaccinated", "people_fully_vaccinated"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date", "case_cnt", "death_cnt"]).copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)


def _build_fit_table(df_cases: pd.DataFrame, train_start: datetime, eval_end: datetime) -> pd.DataFrame:
    out = df_cases[(df_cases["date"] >= train_start) & (df_cases["date"] <= eval_end)].copy()
    out["day_since100"] = (out["date"] - train_start).dt.days.astype(int)
    return out[
        [
            "date",
            "day_since100",
            "case_cnt",
            "death_cnt",
            "total_hospitalization",
            "people_vaccinated",
            "people_fully_vaccinated",
        ]
    ].reset_index(drop=True)


def _params_tuple_from_row(row: pd.Series) -> tuple:
    return (
        float(row["Infection Rate"]),
        float(row["Median Day of Action"]),
        float(row["Rate of Action"]),
        float(row["Rate of Death"]),
        float(row["Mortality Rate"]),
        float(row["Rate of Mortality Rate Decay"]),
        float(row["Internal Parameter 1"]),
        float(row["Internal Parameter 2"]),
        float(row["Jump Magnitude"]),
        float(row["Jump Time"]),
        float(row["Jump Decay"]),
    )


# DELPHI parameter-box boundaries (see default_bounds_params in pandemic_params.py).
# When the train fit lands on these corners it usually means the optimiser
# failed to find a meaningful fit; feeding such a fit into THEMIS tends to produce
# runaway trajectories because alpha*gamma is anchored at a non-physical value.
_ALPHA_UB = 1.25
_ALPHA_LB = 0.75
_R_S_LB = 1.0
_R_S_UB = 3.0
_DOA_UB = 10.0
_DOA_LB = -10.0
_BOUND_EPS = 1e-3


def _fit_quality_ok(
    params_row: pd.Series,
    max_train_mape: float,
    max_corners: int,
) -> Tuple[bool, str]:
    """Return (ok, reason) based on whether the train-window DELPHI fit looks usable.

    A fit is rejected if: (a) train MAPE above threshold, (b) too many of the core
    parameters are stuck at a box corner, or (c) the MAPE is non-finite.
    """
    alpha = float(params_row["Infection Rate"])
    r_s = float(params_row["Rate of Action"])
    doa = float(params_row["Median Day of Action"])
    train_mape = float(params_row.get("MAPE", np.nan))

    at_alpha_ub = alpha >= _ALPHA_UB - _BOUND_EPS
    at_alpha_lb = alpha <= _ALPHA_LB + _BOUND_EPS
    at_rs_lb = r_s <= _R_S_LB + _BOUND_EPS
    at_rs_ub = r_s >= _R_S_UB - _BOUND_EPS
    at_doa_ub = doa >= _DOA_UB - _BOUND_EPS
    at_doa_lb = doa <= _DOA_LB + _BOUND_EPS

    corners_hit = (
        int(at_alpha_ub) + int(at_alpha_lb)
        + int(at_rs_lb) + int(at_rs_ub)
        + int(at_doa_ub) + int(at_doa_lb)
    )

    if not np.isfinite(train_mape):
        return False, f"nonfinite_train_mape(alpha={alpha:.2f},r_s={r_s:.2f},doa={doa:.2f})"
    if train_mape > max_train_mape:
        return False, (
            f"train_mape_too_high({train_mape:.1f}%>{max_train_mape:.1f}%,"
            f"alpha={alpha:.2f},r_s={r_s:.2f},doa={doa:.2f})"
        )
    if corners_hit > max_corners:
        corner_tags = []
        if at_alpha_ub:
            corner_tags.append("alpha@UB")
        if at_alpha_lb:
            corner_tags.append("alpha@LB")
        if at_rs_lb:
            corner_tags.append("r_s@LB")
        if at_rs_ub:
            corner_tags.append("r_s@UB")
        if at_doa_ub:
            corner_tags.append("DoA@UB")
        if at_doa_lb:
            corner_tags.append("DoA@LB")
        return False, (
            f"too_many_corners({corners_hit}>{max_corners}:" + ",".join(corner_tags)
            + f",mape={train_mape:.1f})"
        )
    return True, ""


def _arctan_gamma_at(day_idx: float, params_tuple: tuple) -> float:
    """Evaluate DELPHI's arctan-gaussian gamma at day_idx (days since train_start)."""
    _, days, r_s, _, _, _, _, _, jump, t_jump, std = params_tuple
    t = float(day_idx)
    return float(
        (2.0 / np.pi) * np.arctan(-(t - days) / 20.0 * r_s)
        + 1.0
        + jump * np.exp(-((t - t_jump) ** 2) / (2.0 * std ** 2))
    )


def _fit_delphi_window(
    region: str,
    fit_table: pd.DataFrame,
    train_end: datetime,
    eval_end: datetime,
    optimization_method: str = "tnc",
) -> FitOutput:
    yesterday = (train_end - timedelta(days=1)).strftime("%Y%m%d")
    df_params, _, df_pred_since_100, _ = solve_and_predict_area(
        region=region,
        yesterday=yesterday,
        past_parameters=None,
        totalcases=fit_table.copy(),
        end_date=eval_end.strftime("%Y-%m-%d"),
        optimization_method=optimization_method,
    )
    params_row = df_params.iloc[0].copy()
    params_tuple = _params_tuple_from_row(params_row)
    pred = df_pred_since_100.copy()
    pred["date"] = pd.to_datetime(pred["Day"], errors="coerce").dt.normalize()
    pred["pred_cases"] = pd.to_numeric(pred["Total Detected"], errors="coerce")
    pred["pred_deaths"] = pd.to_numeric(pred["Total Detected Deaths"], errors="coerce")
    pred = pred.dropna(subset=["date", "pred_cases", "pred_deaths"]).copy()
    pred = pred[["date", "pred_cases", "pred_deaths"]].sort_values("date").reset_index(drop=True)
    return FitOutput(params_row=params_row, preds_df=pred, params_tuple=params_tuple)


def _predict_delphi_past_params(
    region: str,
    cases_df: pd.DataFrame,
    population: float,
    eval_end: datetime,
) -> Optional[Tuple[FitOutput, datetime]]:
    """Run DELPHI forward using pre-fitted past_parameters (no optimisation).

    Uses each region's own ``Data Start Date`` as the ODE origin so that
    ``Median Day of Action``, ``Jump Time`` etc. are on the correct time-scale.

    Returns ``(FitOutput, data_start_date)`` or ``None`` if the region is
    not found in ``past_parameters``.
    """
    country, province = region_symbol_country_dict[region]
    match = _past_parameters_df[
        (_past_parameters_df.Country == country)
        & (_past_parameters_df.Province == province)
    ]
    if len(match) == 0:
        return None

    prow = match.iloc[-1]
    data_start_date = pd.to_datetime(prow["Data Start Date"]).normalize().to_pydatetime()

    params_tuple = _params_tuple_from_row(prow)

    init_row = cases_df[cases_df["date"] >= data_start_date].head(1)
    if init_row.empty:
        return None
    initial_cases = float(init_row.iloc[0]["case_cnt"])
    initial_deaths = float(init_row.iloc[0]["death_cnt"])

    pred = _simulate_delphi_with_windows(
        params=params_tuple,
        population=population,
        initial_cases=initial_cases,
        initial_deaths=initial_deaths,
        train_start=data_start_date,
        eval_end=eval_end,
        policy_windows=None,
        policy_start_t=None,
    )
    if pred is None:
        return None

    return FitOutput(params_row=prow.copy(), preds_df=pred, params_tuple=params_tuple), data_start_date


def _read_policy_data(country: str, province: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    try:
        if country == "US" and province != "None":
            pol = read_policy_data_us_only(
                state=province,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
        else:
            policy_country = _CASE_TO_POLICY_COUNTRY.get(country, country)
            pol = read_oxford_country_policy_data(
                country=policy_country,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
    except Exception:
        return None
    if pol is None or pol.empty:
        return None
    out = pol.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    for p in future_policies:
        if p not in out.columns:
            out[p] = 0
        out[p] = out[p].fillna(0).astype(int)
    return out[["date"] + future_policies].reset_index(drop=True)


def _active_policy_from_row(row: pd.Series) -> Optional[str]:
    for p in future_policies:
        if int(row.get(p, 0)) == 1:
            return p
    return None


def _estimate_region_policy_gammas(
    fit_params: tuple,
    train_start: datetime,
    policy_train_df: pd.DataFrame,
    policy_days_thresh: int,
) -> Tuple[Dict[str, float], Dict[str, int], float]:
    default_gammas = dict(sorted(default_dict_normalized_policy_gamma.items(), key=lambda x: x[0]))
    _, days, r_s, _, _, _, _, _, jump, t_jump, std = fit_params
    params_list = [train_start, days, r_s, jump, t_jump, std]
    work = policy_train_df.copy()
    work["Gamma"] = [gamma_t(day.to_pydatetime(), params_list) for day in work["date"]]

    region_gamma_all = {p: float(work.loc[work[p] == 1, "Gamma"].mean()) for p in future_policies}
    region_policy_counts = {p: int(work[p].sum()) for p in future_policies}

    policy_t = [_active_policy_from_row(row) for _, row in work.iterrows()]
    valid_mask = np.array(
        [
            (pt is not None) and (region_policy_counts.get(pt, 0) > policy_days_thresh)
            for pt in policy_t
        ],
        dtype=bool,
    )

    if not valid_mask.any():
        return default_gammas, region_policy_counts, 1.0

    default_gamma = np.array(
        [default_gammas[pt] for pt, keep in zip(policy_t, valid_mask) if keep],
        dtype=float,
    )
    z_vals = work.loc[valid_mask, "Gamma"].to_numpy(dtype=float) / np.maximum(default_gamma, 1e-8)
    z_mean = float(np.mean(z_vals))

    region_gamma = {}
    for p in future_policies:
        if region_policy_counts.get(p, 0) > policy_days_thresh and np.isfinite(region_gamma_all[p]):
            region_gamma[p] = min(float(region_gamma_all[p]), 1.0)
        else:
            region_gamma[p] = min(float(default_gammas[p] * z_mean), 1.0)
    return region_gamma, region_policy_counts, z_mean



def _simulate_delphi_with_windows(
    params: tuple,
    population: float,
    initial_cases: float,
    initial_deaths: float,
    train_start: datetime,
    eval_end: datetime,
    policy_windows: Optional[Dict[Tuple[float, float], float]],
    policy_start_t: Optional[float],
) -> Optional[pd.DataFrame]:
    pop_r = float(initial_deaths * 5.0)
    x0 = get_initial_conditions(
        params_fitted=params,
        global_params_fixed=(population, pop_r, initial_deaths, initial_cases, p_v, p_d, p_h),
    )
    max_t = int((eval_end - train_start).days) + 1
    if max_t <= 1:
        return None

    t_eval = np.arange(0, max_t, dtype=float)

    def rhs(t, x, alpha, days, r_s, r_dth, p_dth, r_decay, k1, k2, jump, t_jump, std):
        return model_covid(
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
            population,
            policy_windows,
            policy_start_t,
            max_t,
        )

    try:
        sol = solve_ivp(
            fun=rhs,
            y0=x0,
            t_span=[0, float(max_t - 1)],
            t_eval=t_eval,
            args=params,
            rtol=1e-6,
            atol=1e-6,
        )
    except Exception:
        return None
    if (not sol.success) or sol.y.shape[1] != max_t:
        return None

    dates = [train_start + timedelta(days=int(i)) for i in range(max_t)]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "pred_cases": sol.y[15, :].astype(float),
            "pred_deaths": sol.y[14, :].astype(float),
        }
    )
    return out


def _fit_predict_seir(
    cases_df: pd.DataFrame,
    population: float,
    train_start: datetime,
    train_end: datetime,
    eval_end: datetime,
    min_train_days: int,
) -> Optional[pd.DataFrame]:
    train = cases_df[(cases_df["date"] >= train_start) & (cases_df["date"] <= train_end)].copy()
    if train["date"].nunique() < min_train_days:
        return None

    t_train = (train["date"] - train_start).dt.days.to_numpy(dtype=float)
    y_cases = train["case_cnt"].to_numpy(dtype=float)
    y_deaths = train["death_cnt"].to_numpy(dtype=float)
    if len(t_train) < 2:
        return None

    cases0 = float(y_cases[0])
    deaths0 = float(y_deaths[0])
    max_day = int((eval_end - train_start).days)
    if max_day < 1:
        return None
    t_eval_full = np.arange(0, max_day + 1, dtype=float)
    sigma = 1.0 / 5.2
    gamma_rec = 1.0 / 10.0

    def solve_theta(theta: np.ndarray, t_eval: np.ndarray):
        beta, mu, detect, e0, i0 = theta
        e0 = max(float(e0), 1.0)
        i0 = max(float(i0), 1.0)
        d0 = max(float(deaths0), 0.0)
        s0 = population - e0 - i0 - d0
        if s0 <= 1.0:
            return None
        x0 = [s0, e0, i0, 0.0, d0, max(float(cases0), 1.0)]

        def rhs(t, x, beta_param, mu_param, detect_param):
            s, e, i, r, d, cdet = x
            new_exposed = beta_param * s * i / population
            ds = -new_exposed
            de = new_exposed - sigma * e
            di = sigma * e - gamma_rec * i - mu_param * i
            dr = gamma_rec * i
            dd = mu_param * i
            dc = detect_param * sigma * e
            return [ds, de, di, dr, dd, dc]

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
    upper_pop = max(population * 0.2, 1e4)
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
        options={"maxiter": 150, "ftol": 1e-6},
    )
    best_theta = res.x if np.all(np.isfinite(res.x)) else init
    sol_full = solve_theta(best_theta, t_eval_full)
    if sol_full is None:
        return None

    dates = [train_start + timedelta(days=int(i)) for i in range(len(t_eval_full))]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "pred_cases": sol_full.y[5, :].astype(float),
            "pred_deaths": sol_full.y[4, :].astype(float),
        }
    )
    return out


def _fit_predict_sir(
    cases_df: pd.DataFrame,
    population: float,
    train_start: datetime,
    train_end: datetime,
    eval_end: datetime,
    min_train_days: int,
) -> Optional[pd.DataFrame]:
    """SIR model: S -> I -> R.  No exposed compartment, no death compartment."""
    train = cases_df[(cases_df["date"] >= train_start) & (cases_df["date"] <= train_end)].copy()
    if train["date"].nunique() < min_train_days:
        return None
    t_train = (train["date"] - train_start).dt.days.to_numpy(dtype=float)
    y_cases = train["case_cnt"].to_numpy(dtype=float)
    y_deaths = train["death_cnt"].to_numpy(dtype=float)
    if len(t_train) < 2:
        return None

    cases0 = float(y_cases[0])
    deaths0 = float(y_deaths[0])
    max_day = int((eval_end - train_start).days)
    if max_day < 1:
        return None
    t_eval_full = np.arange(0, max_day + 1, dtype=float)
    gamma_rec = 1.0 / 14.0

    def solve_theta(theta, t_eval):
        beta, detect, i0 = theta
        i0 = max(float(i0), 1.0)
        r0 = max(float(cases0) * 0.1, 0.0)
        s0 = population - i0 - r0
        if s0 <= 1.0:
            return None
        x0 = [s0, i0, r0, max(float(cases0), 1.0)]

        def rhs(t, x, b, det):
            s, i, r, cdet = x
            new_inf = b * s * i / population
            return [-new_inf, new_inf - gamma_rec * i, gamma_rec * i, det * new_inf]

        try:
            sol = solve_ivp(rhs, y0=x0, t_span=[0, float(t_eval[-1])],
                            t_eval=t_eval, args=(beta, detect), rtol=1e-5, atol=1e-5)
        except Exception:
            return None
        return sol if sol.success else None

    i0_guess = max(cases0 * 0.3, 10.0)
    init = np.array([0.5, 0.3, i0_guess], dtype=float)
    upper_pop = max(population * 0.2, 1e4)
    bounds = [(0.05, 3.5), (0.02, 1.0), (1.0, upper_pop)]

    def objective(theta):
        sol = solve_theta(theta, t_train)
        if sol is None:
            return 1e12
        w = np.linspace(1.0, 2.0, len(t_train))
        err = (np.log1p(sol.y[3, :]) - np.log1p(y_cases)) ** 2
        return float(np.mean(w * err))

    res = minimize(objective, x0=init, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 150, "ftol": 1e-6})
    best = res.x if np.all(np.isfinite(res.x)) else init
    sol_full = solve_theta(best, t_eval_full)
    if sol_full is None:
        return None
    dates = [train_start + timedelta(days=int(i)) for i in range(len(t_eval_full))]
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "pred_cases": sol_full.y[3, :].astype(float),
        "pred_deaths": np.full(len(dates), float(deaths0)),
    })


def _fit_predict_seird(
    cases_df: pd.DataFrame,
    population: float,
    train_start: datetime,
    train_end: datetime,
    eval_end: datetime,
    min_train_days: int,
) -> Optional[pd.DataFrame]:
    """SEIRD model: S -> E -> I -> R or D.  Explicit death compartment."""
    train = cases_df[(cases_df["date"] >= train_start) & (cases_df["date"] <= train_end)].copy()
    if train["date"].nunique() < min_train_days:
        return None
    t_train = (train["date"] - train_start).dt.days.to_numpy(dtype=float)
    y_cases = train["case_cnt"].to_numpy(dtype=float)
    y_deaths = train["death_cnt"].to_numpy(dtype=float)
    if len(t_train) < 2:
        return None

    cases0 = float(y_cases[0])
    deaths0 = float(y_deaths[0])
    max_day = int((eval_end - train_start).days)
    if max_day < 1:
        return None
    t_eval_full = np.arange(0, max_day + 1, dtype=float)
    sigma = 1.0 / 5.2

    def solve_theta(theta, t_eval):
        beta, gamma_r, mu, detect, e0, i0 = theta
        e0 = max(float(e0), 1.0)
        i0 = max(float(i0), 1.0)
        d0 = max(float(deaths0), 0.0)
        s0 = population - e0 - i0 - d0
        if s0 <= 1.0:
            return None
        x0 = [s0, e0, i0, 0.0, d0, max(float(cases0), 1.0)]

        def rhs(t, x, b, gr, m, det):
            s, e, i, r, d, cdet = x
            new_exp = b * s * i / population
            return [
                -new_exp,
                new_exp - sigma * e,
                sigma * e - gr * i - m * i,
                gr * i,
                m * i,
                det * sigma * e,
            ]

        try:
            sol = solve_ivp(rhs, y0=x0, t_span=[0, float(t_eval[-1])],
                            t_eval=t_eval, args=(beta, gamma_r, mu, detect),
                            rtol=1e-5, atol=1e-5)
        except Exception:
            return None
        return sol if sol.success else None

    e0_guess = max(cases0 * 0.5, 10.0)
    i0_guess = max(cases0 * 0.25, 10.0)
    init = np.array([0.8, 0.1, 0.02, 0.25, e0_guess, i0_guess], dtype=float)
    upper_pop = max(population * 0.2, 1e4)
    bounds = [(0.05, 3.5), (0.02, 0.5), (1e-4, 0.4), (0.02, 1.0),
              (1.0, upper_pop), (1.0, upper_pop)]

    def objective(theta):
        sol = solve_theta(theta, t_train)
        if sol is None:
            return 1e12
        w = np.linspace(1.0, 2.0, len(t_train))
        err_c = (np.log1p(sol.y[5, :]) - np.log1p(y_cases)) ** 2
        err_d = (np.log1p(sol.y[4, :]) - np.log1p(y_deaths)) ** 2
        return float(np.mean(w * (err_c + err_d)))

    res = minimize(objective, x0=init, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 150, "ftol": 1e-6})
    best = res.x if np.all(np.isfinite(res.x)) else init
    sol_full = solve_theta(best, t_eval_full)
    if sol_full is None:
        return None
    dates = [train_start + timedelta(days=int(i)) for i in range(len(t_eval_full))]
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "pred_cases": sol_full.y[5, :].astype(float),
        "pred_deaths": sol_full.y[4, :].astype(float),
    })


def _compute_metrics(
    model_name: str,
    region: RegionContext,
    pred_df: pd.DataFrame,
    eval_start: datetime,
    eval_end: datetime,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    truth = region.cases[(region.cases["date"] >= eval_start) & (region.cases["date"] <= eval_end)][
        ["date", "case_cnt", "death_cnt"]
    ].copy()
    pred = pred_df[(pred_df["date"] >= eval_start) & (pred_df["date"] <= eval_end)][
        ["date", "pred_cases", "pred_deaths"]
    ].copy()
    merged = truth.merge(pred, on="date", how="inner")
    merged["ape_cases_pct"] = [
        _safe_ape(p, a) for p, a in zip(merged["pred_cases"], merged["case_cnt"])
    ]
    merged["ape_deaths_pct"] = [
        _safe_ape(p, a) for p, a in zip(merged["pred_deaths"], merged["death_cnt"])
    ]
    merged["region"] = region.region
    merged["country"] = region.country
    merged["province"] = region.province
    merged["model"] = model_name
    summary = {
        "region": region.region,
        "country": region.country,
        "province": region.province,
        "model": model_name,
        "n_eval_days": int(merged["date"].nunique()),
        "mean_daily_ape_cases_pct": float(np.nanmean(merged["ape_cases_pct"])),
        "mean_daily_ape_deaths_pct": float(np.nanmean(merged["ape_deaths_pct"])),
        "mean_daily_ape_both_pct": float(
            np.nanmean(
                [
                    float(np.nanmean(merged["ape_cases_pct"])),
                    float(np.nanmean(merged["ape_deaths_pct"])),
                ]
            )
        ),
    }
    return merged, summary


def _bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_lower, ci_upper) via bootstrap percentile method."""
    if rng is None:
        rng = np.random.default_rng(42)
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return (np.nan, np.nan, np.nan)
    point = float(stat_fn(clean))
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(clean, size=len(clean), replace=True)
        boot_stats[i] = stat_fn(sample)
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (point, lo, hi)


def _aggregate_model_metrics(
    region_metrics: pd.DataFrame,
    n_boot: int = 5000,
    ci_alpha: float = 0.05,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: List[dict] = []
    for model, g in region_metrics.groupby("model"):
        row: dict = {"model": model, "regions": int(g["region"].nunique())}

        for metric_col, label in [
            ("mean_daily_ape_cases_pct", "cases"),
            ("mean_daily_ape_deaths_pct", "deaths"),
            ("mean_daily_ape_both_pct", "both"),
        ]:
            vals = g[metric_col].to_numpy(dtype=float)

            mean_pt, mean_lo, mean_hi = _bootstrap_ci(vals, np.nanmean, n_boot, ci_alpha, rng)
            row[f"mean_regions_{label}_pct"] = mean_pt
            row[f"mean_regions_{label}_ci_lo"] = mean_lo
            row[f"mean_regions_{label}_ci_hi"] = mean_hi

            med_pt, med_lo, med_hi = _bootstrap_ci(vals, np.nanmedian, n_boot, ci_alpha, rng)
            row[f"median_regions_{label}_pct"] = med_pt
            row[f"median_regions_{label}_ci_lo"] = med_lo
            row[f"median_regions_{label}_ci_hi"] = med_hi

        rows.append(row)
    out = pd.DataFrame(rows)
    out["rank_mean_cases"] = out["mean_regions_cases_pct"].rank(method="dense", ascending=True)
    out["rank_median_cases"] = out["median_regions_cases_pct"].rank(method="dense", ascending=True)
    return out.sort_values("rank_mean_cases").reset_index(drop=True)


def _check_expected_order(agg: pd.DataFrame) -> Dict[str, object]:
    expected = MODEL_ORDER
    tmp = agg.copy()
    tmp = tmp.sort_values("mean_regions_cases_pct")
    observed = tmp["model"].tolist()
    return {
        "expected_best_to_worst_cases": expected,
        "observed_best_to_worst_cases": observed,
        "matches_expected": observed == expected,
    }


def _register_region(region_id: str, country: str, province: str) -> None:
    """Register a region into the global dicts so solve_and_predict_area can find it."""
    if region_id in region_symbol_country_dict:
        return
    region_symbol_country_dict[region_id] = (country, province)
    pop_row = global_populations[
        (global_populations.Country == country) & (global_populations.Province == province)
    ]
    if not pop_row.empty and "Continent" in pop_row.columns:
        continent = str(pop_row.iloc[0]["Continent"])
    else:
        continent = "Unknown"
    region_symbol_continent_dict[region_id] = continent


def _build_region_contexts(
    all_regions: List[Tuple[str, str, Path]],
    train_start: datetime,
    eval_end: datetime,
    selected_regions: str,
    min_cases_threshold: int,
) -> List[RegionContext]:
    pop_lookup = _build_population_lookup()

    selected_set: Optional[set] = None
    if selected_regions.strip():
        selected_set = {r.strip() for r in selected_regions.split(",") if r.strip()}

    contexts: List[RegionContext] = []
    for country, province, case_path in all_regions:
        rid = _region_id(country, province)
        if selected_set is not None and rid not in selected_set:
            continue

        pop_key = (country, province)
        if pop_key not in pop_lookup:
            continue

        cases = _load_cases(case_path)
        in_window = cases[(cases["date"] >= train_start) & (cases["date"] <= eval_end)]
        if in_window.empty:
            continue

        max_cases_in_window = in_window["case_cnt"].max()
        if not np.isfinite(max_cases_in_window) or max_cases_in_window < min_cases_threshold:
            continue

        _register_region(rid, country, province)

        contexts.append(
            RegionContext(
                region=rid,
                country=country,
                province=province,
                population=float(pop_lookup[pop_key]),
                cases=cases,
            )
        )
    return contexts


def main() -> None:
    args = parse_args()
    train_start = _to_date(args.train_start)
    train_end = _to_date(args.train_end)
    eval_end = _to_date(args.eval_end)
    eval_start = _to_date(args.eval_start) if args.eval_start else train_end
    if not (train_start < train_end <= eval_start <= eval_end):
        raise ValueError("Dates must satisfy train_start < train_end <= eval_end.")

    all_regions = _discover_all_regions()
    print(f"[INFO] Discovered {len(all_regions)} case files.")

    # Build target region contexts only (DELPHI fitting is done only for these)
    all_contexts = _build_region_contexts(
        all_regions,
        train_start=train_start,
        eval_end=eval_end,
        selected_regions=args.regions,
        min_cases_threshold=args.min_cases,
    )
    if len(all_contexts) == 0:
        raise RuntimeError("No region has both cases and population in the requested date window.")

    sampled_region_ids: Optional[List[str]] = None
    if args.random_sample and args.random_sample > 0 and len(all_contexts) > args.random_sample:
        rng = np.random.default_rng(args.random_seed)
        idx = rng.choice(len(all_contexts), size=args.random_sample, replace=False)
        all_contexts = [all_contexts[i] for i in sorted(int(j) for j in idx)]
        sampled_region_ids = [c.region for c in all_contexts]
        print(
            f"[INFO] Randomly sampled {len(all_contexts)} regions with seed={args.random_seed} "
            f"(requested {args.random_sample})."
        )

    target_ids = {c.region for c in all_contexts}
    print(f"[INFO] {len(all_contexts)} target regions for evaluation.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"themis_time_transfer_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    region_metric_rows: List[dict] = []
    daily_rows: List[pd.DataFrame] = []
    params_rows_train: List[pd.Series] = []
    params_rows_lookahead: List[pd.Series] = []
    gamma_rows: List[dict] = []
    themis_cap_rows: List[dict] = []
    failed_regions: List[dict] = []

    # ── GLOBAL GAMMA MATRIX ─────────────────────────────────────────────────
    policy_names_sorted = sorted(default_dict_normalized_policy_gamma.keys())
    policy_index = {p: i for i, p in enumerate(policy_names_sorted)}
    n_policies = len(policy_names_sorted)

    if args.fresh_fit_cache:
        cache_dir = Path(args.fresh_fit_cache)
        print(f"\n[GAMMA] Loading pre-computed gamma matrix from {cache_dir} ...")
        gdata = np.load(cache_dir / "gamma_data.npz")
        gamma_matrix = gdata["gamma_matrix"]
        obs_mask = gdata["obs_mask"]
        completed_matrix = gdata["completed_matrix"]
        k_vector = gdata["k_vector"]
        with open(cache_dir / "gamma_region_ids.json") as f:
            gamma_region_ids = json.load(f)
        fresh_params_df = pd.read_csv(cache_dir / "fresh_params.csv")
        fresh_params_lookup: Dict[str, pd.Series] = {}
        for _, row in fresh_params_df.iterrows():
            fresh_params_lookup[row["region"]] = row
        n_gamma_regions = len(gamma_region_ids)
        print(f"[GAMMA] {n_gamma_regions} regions x {n_policies} policies, "
              f"{obs_mask.sum()} observed ({100*obs_mask.sum()/(n_gamma_regions*n_policies):.1f}%)")
        print(f"[ALS] Loaded rank-1 results. k_R range: [{k_vector.min():.4f}, {k_vector.max():.4f}]")
    else:
        GAMMA_MIN_POLICY_DAYS = 10
        print(f"\n[GAMMA] Building global gamma matrix from past_parameters (min_policy_days={GAMMA_MIN_POLICY_DAYS}) ...")
        gamma_matrix, obs_mask, gamma_region_ids, gamma_policy_names = _build_global_gamma_matrix(
            start_date=train_start.strftime("%Y-%m-%d"),
            end_date=train_end.strftime("%Y-%m-%d"),
            min_policy_days=GAMMA_MIN_POLICY_DAYS,
        )
        n_gamma_regions = len(gamma_region_ids)
        print(f"[GAMMA] {n_gamma_regions} regions x {n_policies} policies, "
              f"{obs_mask.sum()} observed ({100*obs_mask.sum()/(n_gamma_regions*n_policies):.1f}%)")
        print(f"[ALS] Running rank-1 imputation ...")
        completed_matrix, k_vector = _als_rank1_imputation(gamma_matrix, obs_mask)
        print(f"[ALS] Completed. k_R range: [{k_vector.min():.4f}, {k_vector.max():.4f}]")
        fresh_params_lookup = None

    # Build lookup: benchmark region ID -> gamma matrix row index
    def _benchmark_rid_to_gamma_rid(country: str, province: str) -> str:
        prov = province if province else "None"
        return f"{country}__{prov}".replace(" ", "_")

    gamma_rid_to_idx = {rid: i for i, rid in enumerate(gamma_region_ids)}

    # ── FIT DELPHI for target regions and simulate ───────────────────────────
    @dataclass
    class TargetResult:
        ctx: "RegionContext"
        fit_train: FitOutput
        fit_lookahead: Optional[FitOutput]
        fit_table: pd.DataFrame
        policy_eval: pd.DataFrame
        gamma_matrix_idx: int
        sim_origin: datetime  # t=0 for DELPHI ODE: Data Start Date (past-params) or train_start (fresh fit)
        origin_cases: float   # case_cnt at sim_origin
        origin_deaths: float  # death_cnt at sim_origin

    target_results: List[TargetResult] = []

    n_total = len(all_contexts)
    mode_label = "past_parameters (no optimisation)" if args.use_past_params else args.optimization_method
    print(f"\n[FIT] Fitting DELPHI for {n_total} target regions (mode: {mode_label}) ...")
    for idx, ctx in enumerate(all_contexts):
        print(f"[INFO] [{idx+1}/{n_total}] Fitting {ctx.region} ({ctx.country}, {ctx.province})")

        gamma_rid = _benchmark_rid_to_gamma_rid(ctx.country, ctx.province)
        if gamma_rid not in gamma_rid_to_idx:
            failed_regions.append({"region": ctx.region, "reason": "not_in_gamma_matrix"})
            continue
        gm_idx = gamma_rid_to_idx[gamma_rid]

        fit_table = _build_fit_table(ctx.cases, train_start=train_start, eval_end=eval_end)
        if fit_table["date"].nunique() < max(14, args.seir_min_train_days):
            failed_regions.append({"region": ctx.region, "reason": "insufficient_days"})
            continue
        required_dates = {train_start, train_end, eval_end}
        if not required_dates.issubset(set(fit_table["date"].tolist())):
            failed_regions.append({"region": ctx.region, "reason": "missing_required_dates"})
            continue

        # ── DELPHI train fit ──────────────────────────────────────────────
        if args.fresh_fit_cache and fresh_params_lookup is not None and ctx.region in fresh_params_lookup:
            cached = fresh_params_lookup[ctx.region]
            params_tuple = (
                float(cached["Infection Rate"]),
                float(cached["Median Day of Action"]),
                float(cached["Rate of Action"]),
                float(cached["Rate of Death"]),
                float(cached["Mortality Rate"]),
                float(cached["Rate of Mortality Rate Decay"]),
                float(cached["Internal Parameter 1"]),
                float(cached["Internal Parameter 2"]),
                float(cached["Jump Magnitude"]),
                float(cached["Jump Time"]),
                float(cached["Jump Decay"]),
            )
            init_at_origin_row = fit_table[fit_table["date"] == train_start].iloc[0]
            pred = _simulate_delphi_with_windows(
                params=params_tuple,
                population=ctx.population,
                initial_cases=float(init_at_origin_row["case_cnt"]),
                initial_deaths=float(init_at_origin_row["death_cnt"]),
                train_start=train_start,
                eval_end=eval_end,
                policy_windows=None,
                policy_start_t=None,
            )
            if pred is None:
                failed_regions.append({"region": ctx.region, "reason": "cached_sim_failed"})
                continue
            fit_train = FitOutput(params_row=cached, preds_df=pred, params_tuple=params_tuple)
            sim_origin = train_start
            origin_cases = float(init_at_origin_row["case_cnt"])
            origin_deaths = float(init_at_origin_row["death_cnt"])
        elif args.use_past_params:
            result = _predict_delphi_past_params(
                region=ctx.region,
                cases_df=ctx.cases,
                population=ctx.population,
                eval_end=eval_end,
            )
            if result is None:
                failed_regions.append({"region": ctx.region, "reason": "not_in_past_parameters"})
                continue
            fit_train, data_start_date = result
            sim_origin = data_start_date

            init_at_origin = ctx.cases[ctx.cases["date"] >= data_start_date].head(1)
            origin_cases = float(init_at_origin.iloc[0]["case_cnt"])
            origin_deaths = float(init_at_origin.iloc[0]["death_cnt"])
        else:
            try:
                fit_train = _fit_delphi_window(
                    region=ctx.region,
                    fit_table=fit_table,
                    train_end=train_end,
                    eval_end=eval_end,
                    optimization_method=args.optimization_method,
                )
            except Exception as exc:
                failed_regions.append({"region": ctx.region, "reason": f"delphi_train_fit_failed: {type(exc).__name__}"})
                continue

            if args.fit_quality_gate:
                ok, reason = _fit_quality_ok(
                    fit_train.params_row,
                    max_train_mape=args.fit_quality_max_train_mape,
                    max_corners=args.fit_quality_max_corners,
                )
                if not ok:
                    failed_regions.append({"region": ctx.region, "reason": f"fit_quality_gate:{reason}"})
                    continue

            sim_origin = train_start
            init_at_origin_row = fit_table[fit_table["date"] == train_start].iloc[0]
            origin_cases = float(init_at_origin_row["case_cnt"])
            origin_deaths = float(init_at_origin_row["death_cnt"])

        # ── DELPHI lookahead fit (always fresh; skip in past-params mode) ─
        fit_lookahead = None
        if not args.use_past_params:
            try:
                fit_lookahead = _fit_delphi_window(
                    region=ctx.region,
                    fit_table=fit_table,
                    train_end=eval_end,
                    eval_end=eval_end,
                    optimization_method=args.optimization_method,
                )
            except Exception as exc:
                failed_regions.append({"region": ctx.region, "reason": f"delphi_lookahead_fit_failed: {type(exc).__name__}"})
                continue

        params_rows_train.append(fit_train.params_row)
        if fit_lookahead is not None:
            params_rows_lookahead.append(fit_lookahead.params_row)

        policy_eval = _read_policy_data(
            country=ctx.country,
            province=ctx.province,
            start_date=eval_start,
            end_date=eval_end,
        )
        if policy_eval is None:
            failed_regions.append({"region": ctx.region, "reason": "policy_data_unavailable"})
            continue

        target_results.append(TargetResult(
            ctx=ctx, fit_train=fit_train, fit_lookahead=fit_lookahead,
            fit_table=fit_table, policy_eval=policy_eval,
            gamma_matrix_idx=gm_idx,
            sim_origin=sim_origin,
            origin_cases=origin_cases,
            origin_deaths=origin_deaths,
        ))

    # ── SIMULATE all models using rank-1 imputed gammas ──────────────────────
    n_target_eval = len(target_results)
    if n_target_eval == 0:
        raise RuntimeError("No target regions survived fitting.")
    print(f"\n[SIM] Simulating {n_target_eval} target regions with rank-1 imputed gammas (from {n_gamma_regions}-region matrix) ...")
    for sim_idx, tr in enumerate(target_results):
        ctx = tr.ctx
        gm_idx = tr.gamma_matrix_idx
        print(f"[INFO] [{sim_idx+1}/{n_target_eval}] Simulating {ctx.region}")

        region_gammas = {p: float(completed_matrix[gm_idx, policy_index[p]]) for p in policy_names_sorted}

        for p in policy_names_sorted:
            gamma_rows.append({
                "region": ctx.region,
                "country": ctx.country,
                "province": ctx.province,
                "policy": p,
                "gamma_region_policy": float(region_gammas[p]),
                "gamma_observed": float(gamma_matrix[gm_idx, policy_index[p]]) if obs_mask[gm_idx, policy_index[p]] else None,
                "is_observed": bool(obs_mask[gm_idx, policy_index[p]]),
                "train_policy_days": 0,
                "k_R": float(k_vector[gm_idx]),
            })

        pred_delphi = tr.fit_train.preds_df.copy()
        pred_lookahead = tr.fit_lookahead.preds_df.copy() if tr.fit_lookahead is not None else None

        # Use sim_origin (Data Start Date for past-params, train_start for fresh fit)
        origin = tr.sim_origin
        _, days, r_s, _, _, _, _, _, jump, t_jump, std = tr.fit_train.params_tuple
        gamma_const = float(gamma_t(eval_start, [origin, days, r_s, jump, t_jump, std]))
        t_eval_start = float((eval_start - origin).days)
        t_eval_end = float((eval_end - origin).days) + 1.0
        const_windows = {(t_eval_start, t_eval_end): gamma_const}
        pred_constant = _simulate_delphi_with_windows(
            params=tr.fit_train.params_tuple,
            population=ctx.population,
            initial_cases=tr.origin_cases,
            initial_deaths=tr.origin_deaths,
            train_start=origin,
            eval_end=eval_end,
            policy_windows=const_windows,
            policy_start_t=t_eval_start,
        )
        if pred_constant is None:
            failed_regions.append({"region": ctx.region, "reason": "delphi_constant_sim_failed"})
            continue

        policy_windows: Dict[Tuple[float, float], float] = {}
        themis_cap_diag = {"capped_days": 0, "total_days": 0, "mean_ratio_before": 0.0}
        for _, row in tr.policy_eval.iterrows():
            p_name = _active_policy_from_row(row)
            if p_name is None:
                p_name = "No_Measure"
            day_idx = float((row["date"].to_pydatetime() - origin).days)
            g_policy = float(region_gammas.get(p_name, region_gammas.get("No_Measure", 0.1)))
            g_eff = g_policy
            if args.themis_gamma_cap_ratio and args.themis_gamma_cap_ratio > 0:
                g_arctan = max(_arctan_gamma_at(day_idx, tr.fit_train.params_tuple), 1e-3)
                g_cap = args.themis_gamma_cap_ratio * g_arctan
                themis_cap_diag["total_days"] += 1
                themis_cap_diag["mean_ratio_before"] += g_policy / g_arctan
                if g_policy > g_cap:
                    g_eff = g_cap
                    themis_cap_diag["capped_days"] += 1
            policy_windows[(day_idx, day_idx + 1.0)] = g_eff
        if themis_cap_diag["total_days"] > 0:
            themis_cap_diag["mean_ratio_before"] /= themis_cap_diag["total_days"]
        themis_cap_rows.append({
            "region": ctx.region, "country": ctx.country, "province": ctx.province,
            "cap_ratio": float(args.themis_gamma_cap_ratio),
            "eval_days": int(themis_cap_diag["total_days"]),
            "capped_days": int(themis_cap_diag["capped_days"]),
            "capped_fraction": (
                float(themis_cap_diag["capped_days"]) / themis_cap_diag["total_days"]
                if themis_cap_diag["total_days"] > 0 else 0.0
            ),
            "mean_policy_over_arctan_ratio": float(themis_cap_diag["mean_ratio_before"]),
        })

        pred_themis = _simulate_delphi_with_windows(
            params=tr.fit_train.params_tuple,
            population=ctx.population,
            initial_cases=tr.origin_cases,
            initial_deaths=tr.origin_deaths,
            train_start=origin,
            eval_end=eval_end,
            policy_windows=policy_windows,
            policy_start_t=t_eval_start,
        )
        if pred_themis is None:
            failed_regions.append({"region": ctx.region, "reason": "themis_sim_failed"})
            continue

        pred_sir = _fit_predict_sir(
            cases_df=tr.fit_table, population=ctx.population,
            train_start=train_start, train_end=train_end, eval_end=eval_end,
            min_train_days=args.seir_min_train_days,
        )
        pred_seir = _fit_predict_seir(
            cases_df=tr.fit_table, population=ctx.population,
            train_start=train_start, train_end=train_end, eval_end=eval_end,
            min_train_days=args.seir_min_train_days,
        )
        pred_seird = _fit_predict_seird(
            cases_df=tr.fit_table, population=ctx.population,
            train_start=train_start, train_end=train_end, eval_end=eval_end,
            min_train_days=args.seir_min_train_days,
        )
        if pred_seir is None and pred_sir is None and pred_seird is None:
            failed_regions.append({"region": ctx.region, "reason": "all_epi_baselines_failed"})
            continue

        model_predictions = {
            "delphi": pred_delphi,
            "delphi_constant": pred_constant,
            "themis": pred_themis,
        }
        if pred_lookahead is not None:
            model_predictions["delphi_lookahead"] = pred_lookahead
        if pred_sir is not None:
            model_predictions["sir"] = pred_sir
        if pred_seir is not None:
            model_predictions["seir"] = pred_seir
        if pred_seird is not None:
            model_predictions["seird"] = pred_seird

        for model_name, pred_df in model_predictions.items():
            daily, summary = _compute_metrics(
                model_name=model_name,
                region=ctx,
                pred_df=pred_df,
                eval_start=eval_start,
                eval_end=eval_end,
            )
            daily_rows.append(daily)
            region_metric_rows.append(summary)

        if len(region_metric_rows) > 0 and (sim_idx + 1) % 10 == 0:
            pd.DataFrame(region_metric_rows).to_csv(
                run_dir / "region_model_metrics_partial.csv", index=False
            )
            pd.DataFrame(failed_regions).to_csv(
                run_dir / "failed_regions_partial.csv", index=False
            )
            n_models = len(MODEL_ORDER)
            print(f"[INFO]   Saved partial results ({len(region_metric_rows)//n_models} regions done)")

    if len(region_metric_rows) == 0:
        raise RuntimeError("No successful regions were evaluated.")

    region_metrics = pd.DataFrame(region_metric_rows)
    daily_metrics = pd.concat(daily_rows, axis=0, ignore_index=True)
    agg = _aggregate_model_metrics(region_metrics, n_boot=args.bootstrap_n, ci_alpha=args.ci_alpha)
    order_check = _check_expected_order(agg)

    if params_rows_train:
        pd.DataFrame(params_rows_train).to_csv(
            run_dir / "delphi_params_train_20200315_20200615.csv", index=False
        )
    if params_rows_lookahead:
        pd.DataFrame(params_rows_lookahead).to_csv(
            run_dir / "delphi_params_lookahead_20200315_20200915.csv", index=False
        )
    pd.DataFrame(gamma_rows).to_csv(run_dir / "region_policy_gammas_train_window.csv", index=False)
    if themis_cap_rows:
        pd.DataFrame(themis_cap_rows).to_csv(run_dir / "themis_gamma_cap_detail.csv", index=False)
    daily_metrics.to_csv(run_dir / "daily_predictions_ape_detail.csv", index=False)
    region_metrics.to_csv(run_dir / "region_model_metrics.csv", index=False)
    agg.to_csv(run_dir / "aggregate_model_metrics.csv", index=False)
    pd.DataFrame(failed_regions).to_csv(run_dir / "failed_regions.csv", index=False)

    failure_reason_counts: Dict[str, int] = {}
    for r in failed_regions:
        reason_key = str(r.get("reason", "")).split(":", 1)[0].split("(", 1)[0]
        failure_reason_counts[reason_key] = failure_reason_counts.get(reason_key, 0) + 1

    run_config = {
        "train_start": train_start.strftime("%Y-%m-%d"),
        "train_end": train_end.strftime("%Y-%m-%d"),
        "eval_start": eval_start.strftime("%Y-%m-%d"),
        "eval_end": eval_end.strftime("%Y-%m-%d"),
        "gamma_policy_days_thresh": int(args.gamma_policy_days_thresh),
        "seir_min_train_days": int(args.seir_min_train_days),
        "optimization_method": "past_parameters" if args.use_past_params else args.optimization_method,
        "use_past_params": bool(args.use_past_params),
        "min_cases_threshold": int(args.min_cases),
        "bootstrap_n": int(args.bootstrap_n),
        "ci_alpha": float(args.ci_alpha),
        "fit_quality_gate": bool(args.fit_quality_gate),
        "fit_quality_max_train_mape": float(args.fit_quality_max_train_mape),
        "fit_quality_max_corners": int(args.fit_quality_max_corners),
        "themis_gamma_cap_ratio": float(args.themis_gamma_cap_ratio),
        "random_sample": int(args.random_sample),
        "random_seed": int(args.random_seed),
        "sampled_region_ids": sampled_region_ids,
        "total_case_files_discovered": len(all_regions),
        "total_regions_in_gamma_matrix": n_gamma_regions,
        "target_region_ids": sorted(target_ids),
        "regions_evaluated": sorted(region_metrics["region"].unique().tolist()),
        "regions_failed": [r["region"] for r in failed_regions],
        "failed_reason_counts": failure_reason_counts,
        "model_order_expected_best_to_worst": MODEL_ORDER,
        "expected_order_check": order_check,
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    ci_pct = int((1 - args.ci_alpha) * 100)
    print(f"\nOutput directory: {run_dir}")
    print(f"Regions evaluated: {len(region_metrics['region'].unique())}")
    print(f"Regions failed:    {len(failed_regions)}")
    if failure_reason_counts:
        print("Failure reasons (by bucket):")
        for k, v in sorted(failure_reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>4d}  {k}")
    if themis_cap_rows:
        cap_df = pd.DataFrame(themis_cap_rows)
        n_any_cap = int((cap_df["capped_days"] > 0).sum())
        print(
            f"THEMIS gamma-cap: active on {n_any_cap}/{len(cap_df)} regions "
            f"(cap_ratio={args.themis_gamma_cap_ratio}, "
            f"mean pre-cap policy/arctan ratio = {cap_df['mean_policy_over_arctan_ratio'].mean():.2f})."
        )
    print(f"\nAggregate model metrics ({ci_pct}% bootstrap CI, n_boot={args.bootstrap_n}):")
    display_cols = [
        "model", "regions",
        "mean_regions_cases_pct", "mean_regions_cases_ci_lo", "mean_regions_cases_ci_hi",
        "median_regions_cases_pct", "median_regions_cases_ci_lo", "median_regions_cases_ci_hi",
        "mean_regions_deaths_pct", "mean_regions_deaths_ci_lo", "mean_regions_deaths_ci_hi",
        "median_regions_deaths_pct", "median_regions_deaths_ci_lo", "median_regions_deaths_ci_hi",
    ]
    print(agg[[c for c in display_cols if c in agg.columns]].to_string(index=False))
    print("\nExpected order check:", order_check)


if __name__ == "__main__":
    main()
