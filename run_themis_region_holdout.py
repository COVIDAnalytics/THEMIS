import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from pandemic_functions.pandemic_params import (
    default_dict_normalized_policy_gamma,
    future_policies,
    region_symbol_continent_dict,
    region_symbol_country_dict,
)
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    get_dominant_policy,
    past_parameters,
    read_oxford_country_policy_data,
    read_policy_data_us_only,
    run_delphi_policy_scenario,
)
from pandemic_functions.delphi_functions.DELPHI_utils import gamma_t
from policy_functions.policy import Policy


GLOBAL_TRUE_DATA_PATH = "pandemic_functions/pandemic_data/Global_DELPHI_predictions_combined.csv"
PARAM_COLS = [
    "Data Start Date",
    "Median Day of Action",
    "Rate of Action",
    "Jump Magnitude",
    "Jump Time",
    "Jump Decay",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Global region holdout evaluation for THEMIS. "
            "Regions are country/province pairs from DELPHI parameters."
        )
    )
    parser.add_argument("--start-date", type=str, default="2020-03-15", help="Window start date (YYYY-MM-DD).")
    parser.add_argument("--months", type=int, default=3, help="Window length in months.")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of random train/test splits.")
    parser.add_argument("--test-size", type=int, default=0, help="Fixed number of test regions per split (0 => use fraction).")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Test fraction when test-size is 0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-policy-days", type=int, default=5, help="Minimum observed policy-days to use direct train mean gamma.")
    parser.add_argument("--min-true-days", type=int, default=30, help="Minimum historical truth days in the window to keep a region.")
    parser.add_argument("--max-regions", type=int, default=0, help="Optional cap on eligible regions for faster debugging.")
    parser.add_argument("--output-dir", type=str, default="simulation_results", help="Output directory.")
    return parser.parse_args()


def _safe_ape(pred: float, actual: float) -> float:
    if actual == 0:
        return np.nan
    return abs(pred - actual) / abs(actual) * 100.0


def _region_id(country: str, province: str) -> str:
    return f"{country}__{province}".replace(" ", "_")


def _active_policy_from_row(row: pd.Series) -> Optional[str]:
    for policy in future_policies:
        if row.get(policy, 0) == 1:
            return policy
    return None


def _read_policy_data(country: str, province: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    try:
        if country == "US":
            df = read_policy_data_us_only(state=province, start_date=start_date, end_date=end_date)
        else:
            df = read_oxford_country_policy_data(country=country, start_date=start_date, end_date=end_date)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    return df if len(df) > 0 else None


def _load_global_true_data() -> pd.DataFrame:
    df = pd.read_csv(
        GLOBAL_TRUE_DATA_PATH,
        keep_default_na=False,
        usecols=["Country", "Province", "Day", "Total Detected True", "Total Detected Deaths True"],
    )
    df["Province"] = df["Province"].fillna("None")
    df["date"] = pd.to_datetime(df["Day"], errors="coerce")
    df["case_cnt"] = pd.to_numeric(df["Total Detected True"], errors="coerce")
    df["death_cnt"] = pd.to_numeric(df["Total Detected Deaths True"], errors="coerce")
    df = df.dropna(subset=["date", "case_cnt", "death_cnt"]).copy()
    df = df.rename(columns={"Country": "country", "Province": "province"})
    return df[["country", "province", "date", "case_cnt", "death_cnt"]].sort_values(
        ["country", "province", "date"]
    )


def _build_totalcases(df_region_truth: pd.DataFrame, data_start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    df = (
        df_region_truth[(df_region_truth["date"] >= data_start_date) & (df_region_truth["date"] <= end_date)]
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .copy()
    )
    if len(df) < 2:
        return None
    df["day_since100"] = (df["date"] - data_start_date).dt.days
    df["total_hospitalization"] = 0
    df["people_vaccinated"] = 0
    df["people_fully_vaccinated"] = 0
    return df[
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


def _build_historical_policy_vector(policy_data: pd.DataFrame, start_date: datetime, months: int) -> Optional[List[str]]:
    policies: List[str] = []
    try:
        for m in range(months):
            dt1 = start_date + relativedelta(months=m)
            dt2 = dt1 + relativedelta(months=1, days=-1)
            policies.append(get_dominant_policy(policy_data, start_date=dt1, end_date=dt2))
    except Exception:
        return None
    return policies


def _compute_actual_counts(totalcases: pd.DataFrame, start_date: datetime, months: int) -> Optional[Tuple[float, float]]:
    end_date = start_date + relativedelta(months=months)
    window = totalcases[(totalcases["date"] >= start_date) & (totalcases["date"] <= end_date)]
    if len(window) < 2:
        return None
    return (
        float(window.iloc[-1]["case_cnt"] - window.iloc[0]["case_cnt"]),
        float(window.iloc[-1]["death_cnt"] - window.iloc[0]["death_cnt"]),
    )


def _compute_gamma_stats(policy_data: pd.DataFrame, params_list: pd.Series) -> Tuple[Dict[str, float], Dict[str, int], float, int]:
    default_gammas = dict(sorted(default_dict_normalized_policy_gamma.items(), key=lambda x: x[0]))
    gamma_sum = {p: 0.0 for p in default_gammas}
    gamma_count = {p: 0 for p in default_gammas}
    z_sum = 0.0
    z_count = 0
    for _, row in policy_data.iterrows():
        policy_name = _active_policy_from_row(row)
        if policy_name is None or policy_name not in default_gammas:
            continue
        g_val = float(gamma_t(row["date"], params_list))
        gamma_sum[policy_name] += g_val
        gamma_count[policy_name] += 1
        z_sum += g_val / default_gammas[policy_name]
        z_count += 1
    return gamma_sum, gamma_count, z_sum, z_count


def _register_region(region_id: str, country: str, province: str, continent: str) -> None:
    region_symbol_country_dict[region_id] = (country, province)
    region_symbol_continent_dict[region_id] = continent


def prepare_global_artifacts(
    start_date: datetime,
    months: int,
    min_true_days: int,
    max_regions: int = 0,
) -> Dict[str, dict]:
    truth_df = _load_global_true_data()
    artifacts: Dict[str, dict] = {}
    params_unique = (
        past_parameters.sort_values(["Country", "Province", "Data Start Date"])
        .drop_duplicates(subset=["Country", "Province"], keep="last")
        .reset_index(drop=True)
    )

    for _, prow in params_unique.iterrows():
        continent = str(prow["Continent"])
        country = str(prow["Country"])
        province = str(prow["Province"])
        data_start_date = pd.to_datetime(prow["Data Start Date"]).to_pydatetime()
        start_date_used = max(start_date, data_start_date)
        end_date_used = start_date_used + relativedelta(months=months)
        region_truth = truth_df[
            (truth_df["country"] == country)
            & (truth_df["province"] == province)
            & (truth_df["date"] >= start_date_used)
            & (truth_df["date"] <= end_date_used)
        ]
        if region_truth["date"].nunique() < min_true_days:
            continue

        totalcases = _build_totalcases(
            df_region_truth=truth_df[(truth_df["country"] == country) & (truth_df["province"] == province)],
            data_start_date=data_start_date,
            end_date=end_date_used,
        )
        if totalcases is None:
            continue

        policy_data = _read_policy_data(
            country=country,
            province=province,
            start_date=start_date_used.strftime("%Y-%m-%d"),
            end_date=end_date_used.strftime("%Y-%m-%d"),
        )
        if policy_data is None or len(policy_data) == 0:
            continue

        policy_vector = _build_historical_policy_vector(policy_data, start_date_used, months)
        if policy_vector is None:
            continue

        actual_counts = _compute_actual_counts(totalcases, start_date_used, months)
        if actual_counts is None:
            continue
        actual_cases, actual_deaths = actual_counts

        gamma_sum, gamma_count, z_sum, z_count = _compute_gamma_stats(policy_data, prow[PARAM_COLS])
        if z_count == 0:
            continue

        region_id = _region_id(country, province)
        _register_region(region_id, country, province, continent)

        artifacts[region_id] = {
            "region_id": region_id,
            "continent": continent,
            "country": country,
            "province": province,
            "start_date_used": start_date_used.strftime("%Y-%m-%d"),
            "months": months,
            "policy_vector": policy_vector,
            "totalcases": totalcases,
            "actual_cases": actual_cases,
            "actual_deaths": actual_deaths,
            "gamma_sum": gamma_sum,
            "gamma_count": gamma_count,
            "z_sum": z_sum,
            "z_count": z_count,
        }

    if max_regions > 0 and len(artifacts) > max_regions:
        region_ids = sorted(artifacts.keys())[:max_regions]
        artifacts = {rid: artifacts[rid] for rid in region_ids}

    return artifacts


def estimate_train_gammas(
    train_region_ids: List[str],
    artifacts: Dict[str, dict],
    min_policy_days: int,
) -> Tuple[Dict[str, float], float, Dict[str, int]]:
    default_gammas = dict(sorted(default_dict_normalized_policy_gamma.items(), key=lambda x: x[0]))
    sum_gamma = {p: 0.0 for p in default_gammas}
    count_gamma = {p: 0 for p in default_gammas}
    z_sum = 0.0
    z_count = 0
    for rid in train_region_ids:
        art = artifacts[rid]
        for p in default_gammas:
            sum_gamma[p] += art["gamma_sum"][p]
            count_gamma[p] += art["gamma_count"][p]
        z_sum += art["z_sum"]
        z_count += art["z_count"]

    z_mean = (z_sum / z_count) if z_count > 0 else 1.0
    train_gammas: Dict[str, float] = {}
    for p, default_gamma in default_gammas.items():
        if count_gamma[p] >= min_policy_days and count_gamma[p] > 0:
            train_gammas[p] = float(sum_gamma[p] / count_gamma[p])
        else:
            train_gammas[p] = float(default_gamma * z_mean)
    return train_gammas, float(z_mean), count_gamma


def run_global_holdout(
    artifacts: Dict[str, dict],
    n_splits: int,
    test_size: int,
    seed: int,
    min_policy_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    region_ids = sorted(artifacts.keys())
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if test_size >= len(region_ids):
        raise ValueError("test_size must be strictly smaller than number of eligible regions")

    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for split_id in range(1, n_splits + 1):
        test_regions = sorted(rng.choice(region_ids, size=test_size, replace=False).tolist())
        train_regions = [r for r in region_ids if r not in set(test_regions)]
        train_gammas, z_mean, policy_counts = estimate_train_gammas(
            train_region_ids=train_regions,
            artifacts=artifacts,
            min_policy_days=min_policy_days,
        )
        for rid in test_regions:
            art = artifacts[rid]
            policy = Policy(
                policy_type="hypothetical",
                start_date=art["start_date_used"],
                policy_vector=art["policy_vector"],
            )
            pred_output = run_delphi_policy_scenario(policy, rid, art["totalcases"], train_gammas)
            pred_cases = float(pred_output[0])
            pred_deaths = float(pred_output[3])
            rows.append(
                {
                    "split_id": split_id,
                    "test_region_id": rid,
                    "country": art["country"],
                    "province": art["province"],
                    "continent": art["continent"],
                    "start_date_used": art["start_date_used"],
                    "months": art["months"],
                    "policy_vector_historical": " | ".join(art["policy_vector"]),
                    "pred_cases": pred_cases,
                    "actual_cases": art["actual_cases"],
                    "abs_error_cases": abs(pred_cases - art["actual_cases"]),
                    "ape_cases_pct": _safe_ape(pred_cases, art["actual_cases"]),
                    "pred_deaths": pred_deaths,
                    "actual_deaths": art["actual_deaths"],
                    "abs_error_deaths": abs(pred_deaths - art["actual_deaths"]),
                    "ape_deaths_pct": _safe_ape(pred_deaths, art["actual_deaths"]),
                    "train_global_zmean": z_mean,
                    "train_policy_counts": str(policy_counts),
                }
            )

    detail_df = pd.DataFrame(rows)
    split_summary = (
        detail_df.groupby("split_id", as_index=False)
        .agg(
            n_test_regions=("test_region_id", "count"),
            mean_ape_cases_pct=("ape_cases_pct", "mean"),
            mean_ape_deaths_pct=("ape_deaths_pct", "mean"),
            mean_abs_error_cases=("abs_error_cases", "mean"),
            mean_abs_error_deaths=("abs_error_deaths", "mean"),
        )
        .sort_values("split_id")
    )
    return detail_df, split_summary


def main() -> None:
    args = parse_args()
    start_dt = pd.to_datetime(args.start_date).to_pydatetime()
    artifacts = prepare_global_artifacts(
        start_date=start_dt,
        months=args.months,
        min_true_days=args.min_true_days,
        max_regions=args.max_regions,
    )
    eligible_regions = sorted(artifacts.keys())
    if len(eligible_regions) < 3:
        raise RuntimeError("Not enough eligible global regions to perform holdout evaluation.")

    if args.test_size > 0:
        test_size = args.test_size
    else:
        test_size = max(1, int(round(len(eligible_regions) * args.test_fraction)))
    test_size = min(test_size, len(eligible_regions) - 1)

    detail_df, split_summary = run_global_holdout(
        artifacts=artifacts,
        n_splits=args.n_splits,
        test_size=test_size,
        seed=args.seed,
        min_policy_days=args.min_policy_days,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = output_dir / f"themis_region_holdout_detail_{ts}.csv"
    summary_path = output_dir / f"themis_region_holdout_summary_{ts}.csv"
    detail_df.to_csv(detail_path, index=False)
    split_summary.to_csv(summary_path, index=False)

    print(f"Eligible global regions: {len(eligible_regions)}")
    print(f"Test regions per split: {test_size}")
    print(f"Saved detail results to: {detail_path}")
    print(f"Saved split summary to: {summary_path}")
    print("Overall mean APE (cases):", round(detail_df["ape_cases_pct"].mean(), 3))
    print("Overall mean APE (deaths):", round(detail_df["ape_deaths_pct"].mean(), 3))
    print("Overall median APE (cases):", round(detail_df["ape_cases_pct"].median(), 3))
    print("Overall median APE (deaths):", round(detail_df["ape_deaths_pct"].median(), 3))


if __name__ == "__main__":
    main()
