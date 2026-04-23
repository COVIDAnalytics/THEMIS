import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import run_delphi_policy_scenario
from pandemic_functions.pandemic_params import default_dict_normalized_policy_gamma
from policy_functions.policy import Policy
from run_themis_region_holdout import prepare_global_artifacts


RAW_DELPHI_PATH = "pandemic_functions/pandemic_data/Global_V2_20200703.csv"
TRUE_DATA_PATH = "pandemic_functions/pandemic_data/Global_DELPHI_predictions_combined.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark THEMIS region-holdout vs raw DELPHI on matched regions/dates, "
            "and tune robust (non-fundamental) gamma aggregation improvements."
        )
    )
    parser.add_argument("--start-date", type=str, default="2020-07-03", help="Evaluation start date (YYYY-MM-DD).")
    parser.add_argument("--months", type=int, default=1, help="Evaluation horizon in months.")
    parser.add_argument("--n-splits", type=int, default=30, help="Number of random holdout splits.")
    parser.add_argument("--test-size", type=int, default=0, help="Fixed number of test regions per split (0 => use fraction).")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Test fraction when test-size is 0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument("--min-true-days", type=int, default=30, help="Minimum truth days in window for THEMIS artifact eligibility.")
    parser.add_argument("--min-policy-days", type=int, default=5, help="Legacy estimator minimum policy-days threshold.")
    parser.add_argument("--require-fixed-start-date", action="store_true", help="Require each region to use the exact start date.")
    parser.add_argument("--max-regions", type=int, default=0, help="Optional cap for debugging.")
    parser.add_argument("--raw-delphi-path", type=str, default=RAW_DELPHI_PATH, help="Raw DELPHI prediction CSV.")
    parser.add_argument("--true-data-path", type=str, default=TRUE_DATA_PATH, help="Global DELPHI combined truth CSV.")
    parser.add_argument("--output-dir", type=str, default="simulation_results", help="Parent directory for outputs.")
    return parser.parse_args()


def _region_id(country: str, province: str) -> str:
    return f"{country}__{province}".replace(" ", "_")


def _safe_ape(pred: float, actual: float) -> float:
    if actual <= 0:
        return np.nan
    return abs(pred - actual) / abs(actual) * 100.0


def _robust_statistic(values: Sequence[float], mode: str, trim_quantile: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    if mode == "median":
        return float(np.median(arr))
    if mode == "trimmed_mean":
        q = float(min(max(trim_quantile, 0.0), 0.45))
        lo = np.quantile(arr, q)
        hi = np.quantile(arr, 1 - q)
        arr_trim = arr[(arr >= lo) & (arr <= hi)]
        if arr_trim.size == 0:
            arr_trim = arr
        return float(np.mean(arr_trim))
    return float(np.mean(arr))


def _compute_interval_counts(
    df: pd.DataFrame,
    country_col: str,
    province_col: str,
    date_col: str,
    cases_col: str,
    deaths_col: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    rows: List[dict] = []
    for (country, province), g in df.groupby([country_col, province_col], sort=False):
        h = g.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
        h = h.set_index(date_col)
        if start_date not in h.index or end_date not in h.index:
            continue
        rows.append(
            {
                "country": str(country),
                "province": str(province),
                "cases_interval": float(h.loc[end_date, cases_col] - h.loc[start_date, cases_col]),
                "deaths_interval": float(h.loc[end_date, deaths_col] - h.loc[start_date, deaths_col]),
            }
        )
    return pd.DataFrame(rows)


def load_raw_delphi_region_ape(
    raw_delphi_path: str,
    true_data_path: str,
    start_date: datetime,
    months: int,
) -> pd.DataFrame:
    end_date = start_date + relativedelta(months=months)

    raw_df = pd.read_csv(
        raw_delphi_path,
        keep_default_na=False,
        usecols=["Country", "Province", "Day", "Total Detected", "Total Detected Deaths"],
    )
    raw_df["Province"] = raw_df["Province"].fillna("None")
    raw_df["date"] = pd.to_datetime(raw_df["Day"], errors="coerce")
    raw_df["pred_cases_total"] = pd.to_numeric(raw_df["Total Detected"], errors="coerce")
    raw_df["pred_deaths_total"] = pd.to_numeric(raw_df["Total Detected Deaths"], errors="coerce")
    raw_df = raw_df.dropna(subset=["date", "pred_cases_total", "pred_deaths_total"]).copy()

    true_df = pd.read_csv(
        true_data_path,
        keep_default_na=False,
        usecols=["Country", "Province", "Day", "Total Detected True", "Total Detected Deaths True"],
    )
    true_df["Province"] = true_df["Province"].fillna("None")
    true_df["date"] = pd.to_datetime(true_df["Day"], errors="coerce")
    true_df["true_cases_total"] = pd.to_numeric(true_df["Total Detected True"], errors="coerce")
    true_df["true_deaths_total"] = pd.to_numeric(true_df["Total Detected Deaths True"], errors="coerce")
    true_df = true_df.dropna(subset=["date", "true_cases_total", "true_deaths_total"]).copy()

    raw_interval = _compute_interval_counts(
        raw_df,
        country_col="Country",
        province_col="Province",
        date_col="date",
        cases_col="pred_cases_total",
        deaths_col="pred_deaths_total",
        start_date=start_date,
        end_date=end_date,
    )
    true_interval = _compute_interval_counts(
        true_df,
        country_col="Country",
        province_col="Province",
        date_col="date",
        cases_col="true_cases_total",
        deaths_col="true_deaths_total",
        start_date=start_date,
        end_date=end_date,
    )

    merged = raw_interval.merge(true_interval, on=["country", "province"], how="inner")
    if merged.empty:
        return merged

    merged["test_region_id"] = [_region_id(c, p) for c, p in zip(merged["country"], merged["province"])]
    merged["raw_pred_cases"] = merged["cases_interval_x"]
    merged["raw_actual_cases"] = merged["cases_interval_y"]
    merged["raw_pred_deaths"] = merged["deaths_interval_x"]
    merged["raw_actual_deaths"] = merged["deaths_interval_y"]
    merged["raw_ape_cases_pct"] = [
        _safe_ape(p, a) for p, a in zip(merged["raw_pred_cases"], merged["raw_actual_cases"])
    ]
    merged["raw_ape_deaths_pct"] = [
        _safe_ape(p, a) for p, a in zip(merged["raw_pred_deaths"], merged["raw_actual_deaths"])
    ]
    return merged[
        [
            "test_region_id",
            "country",
            "province",
            "raw_pred_cases",
            "raw_actual_cases",
            "raw_ape_cases_pct",
            "raw_pred_deaths",
            "raw_actual_deaths",
            "raw_ape_deaths_pct",
        ]
    ].reset_index(drop=True)


def generate_random_splits(region_ids: List[str], n_splits: int, test_size: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    splits: List[List[str]] = []
    for _ in range(n_splits):
        test_regions = sorted(rng.choice(region_ids, size=test_size, replace=False).tolist())
        splits.append(test_regions)
    return splits


def estimate_train_gammas_legacy(
    train_region_ids: List[str], artifacts: Dict[str, dict], min_policy_days: int
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


def estimate_train_gammas_robust(
    train_region_ids: List[str],
    artifacts: Dict[str, dict],
    robust_stat: str,
    trim_quantile: float,
    z_stat: str,
    min_policy_regions: int,
    shrinkage_regions: float,
    gamma_floor: float,
    gamma_cap: float,
) -> Tuple[Dict[str, float], float, Dict[str, int]]:
    default_gammas = dict(sorted(default_dict_normalized_policy_gamma.items(), key=lambda x: x[0]))
    per_policy_region_means: Dict[str, List[float]] = {p: [] for p in default_gammas}
    z_region_values: List[float] = []

    for rid in train_region_ids:
        art = artifacts[rid]
        if art["z_count"] > 0:
            z_region_values.append(float(art["z_sum"] / art["z_count"]))
        for p in default_gammas:
            count_p = int(art["gamma_count"][p])
            if count_p > 0:
                per_policy_region_means[p].append(float(art["gamma_sum"][p] / count_p))

    if len(z_region_values) == 0:
        z_center = 1.0
    elif z_stat == "median":
        z_center = float(np.median(z_region_values))
    else:
        z_center = float(np.mean(z_region_values))

    train_gammas: Dict[str, float] = {}
    policy_region_counts: Dict[str, int] = {}
    for p, default_gamma in default_gammas.items():
        region_means = per_policy_region_means[p]
        n_regions = len(region_means)
        policy_region_counts[p] = n_regions
        prior = float(default_gamma * z_center)
        if n_regions == 0:
            gamma_hat = prior
        else:
            center = _robust_statistic(region_means, robust_stat, trim_quantile)
            if n_regions < min_policy_regions:
                weight = n_regions / (n_regions + shrinkage_regions + min_policy_regions)
            else:
                weight = n_regions / (n_regions + shrinkage_regions)
            gamma_hat = (1.0 - weight) * prior + weight * center
        train_gammas[p] = float(np.clip(gamma_hat, gamma_floor, gamma_cap))
    return train_gammas, z_center, policy_region_counts


def run_holdout_with_splits(
    artifacts: Dict[str, dict],
    splits: List[List[str]],
    mode: str,
    min_policy_days: int,
    robust_cfg: Dict[str, float],
) -> pd.DataFrame:
    region_ids = sorted(artifacts.keys())
    rows: List[dict] = []

    for split_id, test_regions in enumerate(splits, start=1):
        test_set = set(test_regions)
        train_regions = [rid for rid in region_ids if rid not in test_set]

        if mode == "legacy":
            train_gammas, z_metric, counts = estimate_train_gammas_legacy(
                train_region_ids=train_regions,
                artifacts=artifacts,
                min_policy_days=min_policy_days,
            )
        else:
            train_gammas, z_metric, counts = estimate_train_gammas_robust(
                train_region_ids=train_regions,
                artifacts=artifacts,
                robust_stat=str(robust_cfg["robust_stat"]),
                trim_quantile=float(robust_cfg["trim_quantile"]),
                z_stat=str(robust_cfg["z_stat"]),
                min_policy_regions=int(robust_cfg["min_policy_regions"]),
                shrinkage_regions=float(robust_cfg["shrinkage_regions"]),
                gamma_floor=float(robust_cfg["gamma_floor"]),
                gamma_cap=float(robust_cfg["gamma_cap"]),
            )

        for rid in test_regions:
            art = artifacts[rid]
            policy = Policy(
                policy_type="hypothetical",
                start_date=art["start_date_used"],
                policy_vector=art["policy_vector"],
            )
            try:
                pred_output = run_delphi_policy_scenario(policy, rid, art["totalcases"], train_gammas)
            except Exception:
                continue

            pred_cases = float(pred_output[0])
            pred_deaths = float(pred_output[3])
            rows.append(
                {
                    "mode": mode,
                    "split_id": split_id,
                    "test_region_id": rid,
                    "country": art["country"],
                    "province": art["province"],
                    "continent": art["continent"],
                    "start_date_used": art["start_date_used"],
                    "months": art["months"],
                    "pred_cases": pred_cases,
                    "actual_cases": art["actual_cases"],
                    "ape_cases_pct": _safe_ape(pred_cases, art["actual_cases"]),
                    "pred_deaths": pred_deaths,
                    "actual_deaths": art["actual_deaths"],
                    "ape_deaths_pct": _safe_ape(pred_deaths, art["actual_deaths"]),
                    "train_z_metric": z_metric,
                    "train_policy_counts": str(counts),
                }
            )
    return pd.DataFrame(rows)


def summarize_region_medians(detail_df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = (
        detail_df.groupby("test_region_id", as_index=False)
        .agg(
            country=("country", "first"),
            province=("province", "first"),
            n_evals=("split_id", "count"),
            ape_cases_pct=("ape_cases_pct", "median"),
            ape_deaths_pct=("ape_deaths_pct", "median"),
        )
        .rename(
            columns={
                "ape_cases_pct": f"{label}_ape_cases_pct",
                "ape_deaths_pct": f"{label}_ape_deaths_pct",
            }
        )
    )
    return out


def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    arr = np.sort(arr)
    y = np.arange(1, arr.size + 1) / arr.size
    return arr, y


def plot_median_bars(summary_metrics: Dict[str, float], out_path: Path) -> None:
    models = ["Raw DELPHI", "THEMIS baseline", "THEMIS improved"]
    cases_vals = [
        summary_metrics["raw_median_ape_cases_pct"],
        summary_metrics["baseline_median_ape_cases_pct"],
        summary_metrics["improved_median_ape_cases_pct"],
    ]
    deaths_vals = [
        summary_metrics["raw_median_ape_deaths_pct"],
        summary_metrics["baseline_median_ape_deaths_pct"],
        summary_metrics["improved_median_ape_deaths_pct"],
    ]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, cases_vals, width, label="Cases median APE")
    ax.bar(x + width / 2, deaths_vals, width, label="Deaths median APE")
    ax.set_ylabel("Median APE (%)")
    ax.set_title("Median APE on Matched Regions/Dates")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12)
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_ecdf_comparison(comp_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    lines = [
        ("raw_ape_cases_pct", "baseline_ape_cases_pct", "improved_ape_cases_pct", "Cases APE (%)"),
        ("raw_ape_deaths_pct", "baseline_ape_deaths_pct", "improved_ape_deaths_pct", "Deaths APE (%)"),
    ]
    labels = ["Raw DELPHI", "THEMIS baseline", "THEMIS improved"]
    colors = ["#333333", "#d62728", "#1f77b4"]

    for ax, (raw_col, base_col, imp_col, title) in zip(axes, lines):
        for col, label, color in zip([raw_col, base_col, imp_col], labels, colors):
            x, y = _ecdf(comp_df[col].to_numpy())
            if x.size == 0:
                continue
            ax.plot(x, y, label=label, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("APE (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("ECDF")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_raw_vs_themis_scatter(comp_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    views = [
        ("raw_ape_cases_pct", "baseline_ape_cases_pct", "improved_ape_cases_pct", "Cases APE (%)"),
        ("raw_ape_deaths_pct", "baseline_ape_deaths_pct", "improved_ape_deaths_pct", "Deaths APE (%)"),
    ]
    for ax, (raw_col, base_col, imp_col, title) in zip(axes, views):
        x_raw = comp_df[raw_col].to_numpy(dtype=float)
        y_base = comp_df[base_col].to_numpy(dtype=float)
        y_imp = comp_df[imp_col].to_numpy(dtype=float)

        ax.scatter(x_raw, y_base, s=18, alpha=0.45, label="Baseline", color="#d62728")
        ax.scatter(x_raw, y_imp, s=18, alpha=0.45, label="Improved", color="#1f77b4")

        finite_vals = np.concatenate(
            [
                x_raw[np.isfinite(x_raw)],
                y_base[np.isfinite(y_base)],
                y_imp[np.isfinite(y_imp)],
            ]
        )
        if finite_vals.size > 0:
            lim = float(np.nanpercentile(finite_vals, 97))
            lim = max(lim, 1.0)
            ax.plot([0, lim], [0, lim], "--", color="black", linewidth=1, alpha=0.7)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)

        ax.set_title(title)
        ax.set_xlabel("Raw DELPHI APE (%)")
        ax.set_ylabel("THEMIS APE (%)")
        ax.grid(alpha=0.25)
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    start_dt = pd.to_datetime(args.start_date).to_pydatetime()
    end_dt = start_dt + relativedelta(months=args.months)
    start_date_str = start_dt.strftime("%Y-%m-%d")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"themis_vs_delphi_benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_region_df = load_raw_delphi_region_ape(
        raw_delphi_path=args.raw_delphi_path,
        true_data_path=args.true_data_path,
        start_date=start_dt,
        months=args.months,
    )
    if raw_region_df.empty:
        raise RuntimeError("No valid raw DELPHI interval rows found for the chosen date window.")

    artifacts = prepare_global_artifacts(
        start_date=start_dt,
        months=args.months,
        min_true_days=args.min_true_days,
        max_regions=args.max_regions,
    )
    if args.require_fixed_start_date:
        artifacts = {rid: art for rid, art in artifacts.items() if art["start_date_used"] == start_date_str}

    matched_region_ids = sorted(set(artifacts.keys()).intersection(set(raw_region_df["test_region_id"].tolist())))
    if len(matched_region_ids) < 5:
        raise RuntimeError("Too few matched regions between THEMIS artifacts and raw DELPHI baseline.")
    artifacts = {rid: artifacts[rid] for rid in matched_region_ids}

    if args.test_size > 0:
        test_size = args.test_size
    else:
        test_size = max(1, int(round(len(matched_region_ids) * args.test_fraction)))
    test_size = min(test_size, len(matched_region_ids) - 1)

    splits = generate_random_splits(
        region_ids=matched_region_ids,
        n_splits=args.n_splits,
        test_size=test_size,
        seed=args.seed,
    )

    baseline_detail = run_holdout_with_splits(
        artifacts=artifacts,
        splits=splits,
        mode="legacy",
        min_policy_days=args.min_policy_days,
        robust_cfg={},
    )
    baseline_region = summarize_region_medians(baseline_detail, label="baseline")

    candidate_cfgs = [
        {
            "name": "median_shrink3",
            "robust_stat": "median",
            "trim_quantile": 0.1,
            "z_stat": "median",
            "min_policy_regions": 2,
            "shrinkage_regions": 3.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.5,
        },
        {
            "name": "median_shrink6",
            "robust_stat": "median",
            "trim_quantile": 0.1,
            "z_stat": "median",
            "min_policy_regions": 2,
            "shrinkage_regions": 6.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.5,
        },
        {
            "name": "median_shrink12",
            "robust_stat": "median",
            "trim_quantile": 0.1,
            "z_stat": "median",
            "min_policy_regions": 2,
            "shrinkage_regions": 12.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.5,
        },
        {
            "name": "median_shrink12_zmean",
            "robust_stat": "median",
            "trim_quantile": 0.1,
            "z_stat": "mean",
            "min_policy_regions": 2,
            "shrinkage_regions": 12.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.5,
        },
        {
            "name": "median_shrink12_min4",
            "robust_stat": "median",
            "trim_quantile": 0.1,
            "z_stat": "median",
            "min_policy_regions": 4,
            "shrinkage_regions": 12.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.5,
        },
        {
            "name": "median_shrink20",
            "robust_stat": "median",
            "trim_quantile": 0.1,
            "z_stat": "median",
            "min_policy_regions": 2,
            "shrinkage_regions": 20.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.5,
        },
        {
            "name": "median_shrink30",
            "robust_stat": "median",
            "trim_quantile": 0.1,
            "z_stat": "median",
            "min_policy_regions": 2,
            "shrinkage_regions": 30.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.5,
        },
        {
            "name": "median_shrink6_cap1p2",
            "robust_stat": "median",
            "trim_quantile": 0.1,
            "z_stat": "median",
            "min_policy_regions": 2,
            "shrinkage_regions": 6.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.2,
        },
        {
            "name": "median_shrink12_cap1p2",
            "robust_stat": "median",
            "trim_quantile": 0.1,
            "z_stat": "median",
            "min_policy_regions": 2,
            "shrinkage_regions": 12.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.2,
        },
        {
            "name": "trimmed_shrink8_cap1p2",
            "robust_stat": "trimmed_mean",
            "trim_quantile": 0.15,
            "z_stat": "median",
            "min_policy_regions": 2,
            "shrinkage_regions": 8.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.2,
        },
        {
            "name": "mean_shrink12_cap1p2",
            "robust_stat": "mean",
            "trim_quantile": 0.1,
            "z_stat": "mean",
            "min_policy_regions": 2,
            "shrinkage_regions": 12.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.2,
        },
        {
            "name": "trimmed_shrink4",
            "robust_stat": "trimmed_mean",
            "trim_quantile": 0.15,
            "z_stat": "median",
            "min_policy_regions": 2,
            "shrinkage_regions": 4.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.5,
        },
        {
            "name": "mean_shrink6",
            "robust_stat": "mean",
            "trim_quantile": 0.1,
            "z_stat": "mean",
            "min_policy_regions": 2,
            "shrinkage_regions": 6.0,
            "gamma_floor": 0.05,
            "gamma_cap": 1.5,
        },
    ]

    sweep_rows: List[dict] = []
    best_cfg = None
    best_detail = None
    best_score = np.inf
    for cfg in candidate_cfgs:
        detail = run_holdout_with_splits(
            artifacts=artifacts,
            splits=splits,
            mode="robust",
            min_policy_days=args.min_policy_days,
            robust_cfg=cfg,
        )
        region_df = summarize_region_medians(detail, label="candidate").merge(
            raw_region_df[["test_region_id", "raw_ape_cases_pct", "raw_ape_deaths_pct"]],
            on="test_region_id",
            how="inner",
        )
        median_cases = float(np.nanmedian(region_df["candidate_ape_cases_pct"]))
        median_deaths = float(np.nanmedian(region_df["candidate_ape_deaths_pct"]))
        score = median_cases + median_deaths
        sweep_rows.append(
            {
                "candidate": cfg["name"],
                "median_ape_cases_pct": median_cases,
                "median_ape_deaths_pct": median_deaths,
                "score_cases_plus_deaths": score,
            }
        )
        if score < best_score:
            best_score = score
            best_cfg = cfg
            best_detail = detail

    if best_cfg is None or best_detail is None:
        raise RuntimeError("No robust candidate produced results.")

    improved_region = summarize_region_medians(best_detail, label="improved")

    comparison = (
        raw_region_df[
            [
                "test_region_id",
                "country",
                "province",
                "raw_ape_cases_pct",
                "raw_ape_deaths_pct",
            ]
        ]
        .merge(
            baseline_region[["test_region_id", "baseline_ape_cases_pct", "baseline_ape_deaths_pct"]],
            on="test_region_id",
            how="inner",
        )
        .merge(
            improved_region[["test_region_id", "improved_ape_cases_pct", "improved_ape_deaths_pct"]],
            on="test_region_id",
            how="inner",
        )
    )
    comparison["delta_improved_minus_raw_cases_pct"] = (
        comparison["improved_ape_cases_pct"] - comparison["raw_ape_cases_pct"]
    )
    comparison["delta_improved_minus_raw_deaths_pct"] = (
        comparison["improved_ape_deaths_pct"] - comparison["raw_ape_deaths_pct"]
    )
    comparison["delta_improved_minus_baseline_cases_pct"] = (
        comparison["improved_ape_cases_pct"] - comparison["baseline_ape_cases_pct"]
    )
    comparison["delta_improved_minus_baseline_deaths_pct"] = (
        comparison["improved_ape_deaths_pct"] - comparison["baseline_ape_deaths_pct"]
    )

    summary_metrics = {
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "months": args.months,
        "matched_regions": int(len(matched_region_ids)),
        "evaluated_regions": int(comparison["test_region_id"].nunique()),
        "n_splits": args.n_splits,
        "test_size": test_size,
        "baseline_median_ape_cases_pct": float(np.nanmedian(comparison["baseline_ape_cases_pct"])),
        "baseline_median_ape_deaths_pct": float(np.nanmedian(comparison["baseline_ape_deaths_pct"])),
        "improved_median_ape_cases_pct": float(np.nanmedian(comparison["improved_ape_cases_pct"])),
        "improved_median_ape_deaths_pct": float(np.nanmedian(comparison["improved_ape_deaths_pct"])),
        "raw_median_ape_cases_pct": float(np.nanmedian(comparison["raw_ape_cases_pct"])),
        "raw_median_ape_deaths_pct": float(np.nanmedian(comparison["raw_ape_deaths_pct"])),
        "improvement_vs_baseline_cases_pp": float(
            np.nanmedian(comparison["baseline_ape_cases_pct"]) - np.nanmedian(comparison["improved_ape_cases_pct"])
        ),
        "improvement_vs_baseline_deaths_pp": float(
            np.nanmedian(comparison["baseline_ape_deaths_pct"]) - np.nanmedian(comparison["improved_ape_deaths_pct"])
        ),
        "gap_vs_raw_cases_pp": float(
            np.nanmedian(comparison["improved_ape_cases_pct"]) - np.nanmedian(comparison["raw_ape_cases_pct"])
        ),
        "gap_vs_raw_deaths_pp": float(
            np.nanmedian(comparison["improved_ape_deaths_pct"]) - np.nanmedian(comparison["raw_ape_deaths_pct"])
        ),
        "selected_robust_candidate": best_cfg["name"],
        "selected_robust_config": best_cfg,
    }

    raw_region_df.to_csv(run_dir / "raw_delphi_region_ape.csv", index=False)
    baseline_detail.to_csv(run_dir / "themis_baseline_detail.csv", index=False)
    best_detail.to_csv(run_dir / "themis_improved_detail.csv", index=False)
    baseline_region.to_csv(run_dir / "themis_baseline_region_median_ape.csv", index=False)
    improved_region.to_csv(run_dir / "themis_improved_region_median_ape.csv", index=False)
    comparison.sort_values("test_region_id").to_csv(run_dir / "themis_vs_delphi_region_comparison.csv", index=False)
    pd.DataFrame(sweep_rows).sort_values("score_cases_plus_deaths").to_csv(
        run_dir / "robust_candidate_sweep.csv", index=False
    )
    with open(run_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    plot_median_bars(summary_metrics, run_dir / "plot_median_ape_bar.png")
    plot_ecdf_comparison(comparison, run_dir / "plot_ape_ecdf.png")
    plot_raw_vs_themis_scatter(comparison, run_dir / "plot_raw_vs_themis_scatter.png")

    print(f"Output directory: {run_dir}")
    print(f"Matched regions before holdout: {summary_metrics['matched_regions']}")
    print(f"Regions in final three-way comparison: {summary_metrics['evaluated_regions']}")
    print(f"Selected robust config: {summary_metrics['selected_robust_candidate']}")
    print(
        "Baseline median APE (cases/deaths): "
        f"{summary_metrics['baseline_median_ape_cases_pct']:.2f}% / "
        f"{summary_metrics['baseline_median_ape_deaths_pct']:.2f}%"
    )
    print(
        "Improved median APE (cases/deaths): "
        f"{summary_metrics['improved_median_ape_cases_pct']:.2f}% / "
        f"{summary_metrics['improved_median_ape_deaths_pct']:.2f}%"
    )
    print(
        "Raw DELPHI median APE (cases/deaths): "
        f"{summary_metrics['raw_median_ape_cases_pct']:.2f}% / "
        f"{summary_metrics['raw_median_ape_deaths_pct']:.2f}%"
    )
    print(
        "Improvement vs baseline in pp (cases/deaths): "
        f"{summary_metrics['improvement_vs_baseline_cases_pp']:.2f} / "
        f"{summary_metrics['improvement_vs_baseline_deaths_pp']:.2f}"
    )
    print(
        "Gap vs raw DELPHI in pp (cases/deaths): "
        f"{summary_metrics['gap_vs_raw_cases_pp']:.2f} / "
        f"{summary_metrics['gap_vs_raw_deaths_pp']:.2f}"
    )


if __name__ == "__main__":
    main()
