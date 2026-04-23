"""
Temporal placebo and falsification tests for gamma_{R,i} estimates.

Two complementary tests to detect confounding in THEMIS's NPI effect estimates:

1. Split-half temporal stability: For each (region, policy) pair, split the
   observation days into early/late halves and test whether gamma estimates
   are stable. Instability suggests time-varying confounders (voluntary behavior
   adaptation, testing ramp-up, seasonal effects) rather than stable NPI effects.

2. Lag-shifted falsification: Shift policy assignment dates forward by L days
   and re-estimate gamma. If gamma_{R,i} is driven by the NPI itself, lagged
   assignment should degrade the estimate. If gamma is smooth and driven by
   background trends, lagged assignment produces similar values.

Usage:
    python sensitivity_placebo_tests.py
    python sensitivity_placebo_tests.py --lags 0 7 14 21
"""
import argparse
import json
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pandemic_functions.pandemic_params import (
    default_dict_normalized_policy_gamma,
    future_policies,
)
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    past_parameters,
    read_oxford_country_policy_data,
    read_policy_data_us_only,
)
from pandemic_functions.delphi_functions.DELPHI_utils import gamma_t

POLICY_NAMES = sorted(default_dict_normalized_policy_gamma.keys())
PARAM_COLS = [
    "Data Start Date",
    "Median Day of Action",
    "Rate of Action",
    "Jump Magnitude",
    "Jump Time",
    "Jump Decay",
]


def _read_policy_data(country, province, start_date, end_date):
    try:
        if country == "US":
            return read_policy_data_us_only(
                state=province, start_date=start_date, end_date=end_date
            )
        else:
            return read_oxford_country_policy_data(
                country=country, start_date=start_date, end_date=end_date
            )
    except Exception:
        return None


def _active_policy_from_row(row):
    for policy in future_policies:
        if row.get(policy, 0) == 1:
            return policy
    return None


# ---------------------------------------------------------------------------
# Test 1: Split-half temporal stability
# ---------------------------------------------------------------------------

def split_half_gamma_test(
    start_date: str = "2020-03-01",
    end_date: str = "2020-07-31",
    min_days_per_half: int = 7,
) -> pd.DataFrame:
    """
    For each (region, policy) pair, split observation days into early/late halves
    and compute gamma in each half. Return a DataFrame with both estimates.
    """
    params_unique = (
        past_parameters
        .sort_values(["Country", "Province", "Data Start Date"])
        .drop_duplicates(subset=["Country", "Province"], keep="last")
        .reset_index(drop=True)
    )

    rows = []
    for _, prow in params_unique.iterrows():
        country = str(prow["Country"])
        province = str(prow["Province"])
        params_list = prow[PARAM_COLS]

        policy_data = _read_policy_data(country, province, start_date, end_date)
        if policy_data is None or len(policy_data) == 0:
            continue
        policy_data["date"] = pd.to_datetime(policy_data["date"], errors="coerce")
        policy_data = policy_data.dropna(subset=["date"]).reset_index(drop=True)

        for policy_name in POLICY_NAMES:
            if policy_name not in policy_data.columns:
                continue
            active_days = policy_data[policy_data[policy_name] == 1].copy()
            if len(active_days) < 2 * min_days_per_half:
                continue

            active_days = active_days.sort_values("date").reset_index(drop=True)
            gamma_values = np.array([
                float(gamma_t(row["date"], params_list))
                for _, row in active_days.iterrows()
            ])

            mid = len(gamma_values) // 2
            gamma_early = np.mean(gamma_values[:mid])
            gamma_late = np.mean(gamma_values[mid:])
            gamma_full = np.mean(gamma_values)
            n_early = mid
            n_late = len(gamma_values) - mid

            rows.append({
                "region": f"{country}__{province}".replace(" ", "_"),
                "policy": policy_name,
                "gamma_full": gamma_full,
                "gamma_early": gamma_early,
                "gamma_late": gamma_late,
                "n_early": n_early,
                "n_late": n_late,
                "n_total": len(gamma_values),
                "abs_diff": abs(gamma_early - gamma_late),
                "rel_diff": abs(gamma_early - gamma_late) / (gamma_full + 1e-10),
                "drift_direction": "increasing" if gamma_late > gamma_early else "decreasing",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 2: Lag-shifted falsification
# ---------------------------------------------------------------------------

def lag_shifted_gamma_test(
    start_date: str = "2020-03-01",
    end_date: str = "2020-07-31",
    lags: List[int] = None,
    min_policy_days: int = 10,
) -> pd.DataFrame:
    """
    Shift the policy assignment dates by L days and re-compute gamma_{R,i}.
    At lag=0, this is the original estimate. At lag=L, we attribute day t's
    gamma to whatever policy was active at day t-L.
    """
    if lags is None:
        lags = [0, 7, 14, 21]

    params_unique = (
        past_parameters
        .sort_values(["Country", "Province", "Data Start Date"])
        .drop_duplicates(subset=["Country", "Province"], keep="last")
        .reset_index(drop=True)
    )

    rows = []
    for _, prow in params_unique.iterrows():
        country = str(prow["Country"])
        province = str(prow["Province"])
        params_list = prow[PARAM_COLS]
        region_id = f"{country}__{province}".replace(" ", "_")

        policy_data = _read_policy_data(country, province, start_date, end_date)
        if policy_data is None or len(policy_data) == 0:
            continue
        policy_data["date"] = pd.to_datetime(policy_data["date"], errors="coerce")
        policy_data = policy_data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        gamma_series = np.array([
            float(gamma_t(row["date"], params_list))
            for _, row in policy_data.iterrows()
        ])
        policy_data["gamma_val"] = gamma_series

        for lag in lags:
            policy_cols_available = [p for p in POLICY_NAMES if p in policy_data.columns]
            if lag == 0:
                shifted_data = policy_data.copy()
            else:
                shifted_data = policy_data.copy()
                for col in policy_cols_available:
                    shifted_data[col] = shifted_data[col].shift(lag, fill_value=0)
                shifted_data = shifted_data.iloc[lag:].reset_index(drop=True)

            for policy_name in policy_cols_available:
                active = shifted_data[shifted_data[policy_name] == 1]
                if len(active) < min_policy_days:
                    continue
                gamma_mean = active["gamma_val"].mean()
                gamma_std = active["gamma_val"].std()

                rows.append({
                    "region": region_id,
                    "policy": policy_name,
                    "lag": lag,
                    "gamma_mean": gamma_mean,
                    "gamma_std": gamma_std,
                    "n_days": len(active),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis & Plotting
# ---------------------------------------------------------------------------

def analyze_split_half(df: pd.DataFrame) -> dict:
    """Statistical tests on split-half stability."""
    if len(df) == 0:
        return {"error": "No data"}

    diffs = df["gamma_early"] - df["gamma_late"]
    t_stat, p_value_t = stats.ttest_rel(df["gamma_early"], df["gamma_late"])
    w_stat, p_value_w = stats.wilcoxon(diffs, alternative="two-sided")

    icc_result = _compute_icc(df["gamma_early"].values, df["gamma_late"].values)

    pct_increasing = (df["drift_direction"] == "increasing").mean() * 100
    mean_abs_diff = df["abs_diff"].mean()
    mean_rel_diff = df["rel_diff"].mean()
    median_abs_diff = df["abs_diff"].median()

    return {
        "n_pairs": len(df),
        "mean_gamma_early": float(df["gamma_early"].mean()),
        "mean_gamma_late": float(df["gamma_late"].mean()),
        "mean_abs_diff": float(mean_abs_diff),
        "median_abs_diff": float(median_abs_diff),
        "mean_rel_diff_pct": float(mean_rel_diff * 100),
        "pct_drift_increasing": float(pct_increasing),
        "paired_ttest_stat": float(t_stat),
        "paired_ttest_pvalue": float(p_value_t),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(p_value_w),
        "icc": float(icc_result),
    }


def _compute_icc(x, y):
    """ICC(3,1) — two-way mixed, single measures, consistency."""
    n = len(x)
    if n < 3:
        return float("nan")
    data = np.column_stack([x, y])
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ss_rows = 2 * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (2 - 1))
    icc = (ms_rows - ms_error) / (ms_rows + ms_error)
    return icc


def analyze_lag_test(df: pd.DataFrame) -> dict:
    """Analyze how gamma estimates degrade with increasing lag."""
    if len(df) == 0:
        return {"error": "No data"}

    baseline = df[df["lag"] == 0].set_index(["region", "policy"])["gamma_mean"]

    results_per_lag = []
    for lag in sorted(df["lag"].unique()):
        lagged = df[df["lag"] == lag].set_index(["region", "policy"])["gamma_mean"]
        common = baseline.index.intersection(lagged.index)
        if len(common) < 3:
            continue
        corr, p_corr = stats.pearsonr(baseline[common], lagged[common])
        mae = np.mean(np.abs(baseline[common] - lagged[common]))
        rmse = np.sqrt(np.mean((baseline[common] - lagged[common]) ** 2))
        results_per_lag.append({
            "lag": lag,
            "n_pairs": len(common),
            "pearson_corr": float(corr),
            "pearson_pvalue": float(p_corr),
            "mae": float(mae),
            "rmse": float(rmse),
        })

    return results_per_lag


def plot_split_half(df: pd.DataFrame, output_dir: Path):
    """Scatter plot of gamma_early vs gamma_late."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(df["gamma_early"], df["gamma_late"], s=20, alpha=0.5, c="#2c7fb8")
    lims = [
        min(df["gamma_early"].min(), df["gamma_late"].min()) - 0.05,
        max(df["gamma_early"].max(), df["gamma_late"].max()) + 0.05,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect agreement")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("$\\gamma_{R,i}$ (Early Half)", fontsize=12)
    ax.set_ylabel("$\\gamma_{R,i}$ (Late Half)", fontsize=12)
    ax.set_title("Split-Half Temporal Stability of $\\gamma_{R,i}$", fontsize=13)
    ax.legend(fontsize=10)

    ax = axes[1]
    for policy in POLICY_NAMES:
        sub = df[df["policy"] == policy]
        if len(sub) > 0:
            short_name = policy.replace("_", " ")[:25]
            ax.scatter(sub["gamma_early"], sub["gamma_late"], s=20, alpha=0.5, label=short_name)
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("$\\gamma_{R,i}$ (Early Half)", fontsize=12)
    ax.set_ylabel("$\\gamma_{R,i}$ (Late Half)", fontsize=12)
    ax.set_title("Split-Half by Policy Type", fontsize=13)
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    fig.savefig(output_dir / "split_half_scatter.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "split_half_scatter.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_split_half_histogram(df: pd.DataFrame, output_dir: Path):
    """Histogram of early-late differences."""
    fig, ax = plt.subplots(figsize=(8, 5))
    diffs = df["gamma_early"] - df["gamma_late"]
    ax.hist(diffs, bins=30, color="#2c7fb8", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Zero difference")
    ax.axvline(x=diffs.mean(), color="orange", linestyle="-", alpha=0.8,
               label=f"Mean = {diffs.mean():.4f}")
    ax.set_xlabel("$\\gamma^{\\mathrm{early}} - \\gamma^{\\mathrm{late}}$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Early vs. Late $\\gamma_{R,i}$ Differences", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "split_half_histogram.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "split_half_histogram.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_lag_degradation(lag_results: list, output_dir: Path):
    """Plot how correlation and error change with lag."""
    if not lag_results:
        return
    lag_df = pd.DataFrame(lag_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(lag_df["lag"], lag_df["pearson_corr"], "o-", color="#d95f02",
             linewidth=2, markersize=8)
    ax1.set_xlabel("Lag (days)", fontsize=12)
    ax1.set_ylabel("Pearson Correlation with Lag-0", fontsize=12)
    ax1.set_title("Falsification: $\\gamma$ Estimate Degradation with Lag", fontsize=13)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    ax2.plot(lag_df["lag"], lag_df["rmse"], "s-", color="#1b9e77",
             linewidth=2, markersize=8, label="RMSE")
    ax2.plot(lag_df["lag"], lag_df["mae"], "^-", color="#7570b3",
             linewidth=2, markersize=8, label="MAE")
    ax2.set_xlabel("Lag (days)", fontsize=12)
    ax2.set_ylabel("Error vs. Lag-0 Estimate", fontsize=12)
    ax2.set_title("Error Growth with Misattributed Policy Timing", fontsize=13)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "lag_falsification.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "lag_falsification.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Placebo/falsification tests for gamma")
    parser.add_argument("--start-date", default="2020-03-01")
    parser.add_argument("--end-date", default="2020-07-31")
    parser.add_argument("--min-days-per-half", type=int, default=7)
    parser.add_argument("--min-policy-days", type=int, default=10)
    parser.add_argument("--lags", nargs="+", type=int, default=[0, 7, 14, 21])
    parser.add_argument("--output-dir", default="simulation_results/placebo_tests")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TEMPORAL PLACEBO AND FALSIFICATION TESTS")
    print("=" * 70)

    # --- Test 1: Split-half ---
    print("\n[Test 1] Split-half temporal stability ...")
    split_df = split_half_gamma_test(
        start_date=args.start_date,
        end_date=args.end_date,
        min_days_per_half=args.min_days_per_half,
    )
    print(f"  Total (region, policy) pairs: {len(split_df)}")

    if len(split_df) > 0:
        split_df.to_csv(output_dir / "split_half_data.csv", index=False)

        analysis = analyze_split_half(split_df)
        print(f"  Mean |gamma_early - gamma_late|: {analysis['mean_abs_diff']:.4f}")
        print(f"  Mean relative diff: {analysis['mean_rel_diff_pct']:.1f}%")
        print(f"  Paired t-test: t={analysis['paired_ttest_stat']:.3f}, "
              f"p={analysis['paired_ttest_pvalue']:.4f}")
        print(f"  Wilcoxon test: W={analysis['wilcoxon_stat']:.1f}, "
              f"p={analysis['wilcoxon_pvalue']:.4f}")
        print(f"  ICC(3,1): {analysis['icc']:.4f}")
        print(f"  % drifting upward: {analysis['pct_drift_increasing']:.1f}%")

        per_policy = split_df.groupby("policy").agg(
            n=("abs_diff", "count"),
            mean_abs_diff=("abs_diff", "mean"),
            mean_rel_diff=("rel_diff", lambda x: x.mean() * 100),
            mean_gamma=("gamma_full", "mean"),
        ).reset_index()
        print(f"\n  Per-policy stability:\n{per_policy.to_string(index=False)}")
        per_policy.to_csv(output_dir / "split_half_per_policy.csv", index=False)

        plot_split_half(split_df, output_dir)
        plot_split_half_histogram(split_df, output_dir)

        interpretation = ""
        if analysis["icc"] > 0.75:
            interpretation = "STRONG temporal stability (ICC > 0.75): gamma estimates are reliable across time sub-periods, suggesting limited time-varying confounding."
        elif analysis["icc"] > 0.50:
            interpretation = "MODERATE temporal stability (0.50 < ICC < 0.75): some drift detected but dominant signal is stable."
        else:
            interpretation = "WEAK temporal stability (ICC < 0.50): substantial drift detected, suggesting potential time-varying confounding."
        analysis["interpretation"] = interpretation
        print(f"\n  >>> {interpretation}")
    else:
        analysis = {"error": "No valid pairs found"}

    # --- Test 2: Lag-shifted falsification ---
    print(f"\n[Test 2] Lag-shifted falsification (lags={args.lags}) ...")
    lag_df = lag_shifted_gamma_test(
        start_date=args.start_date,
        end_date=args.end_date,
        lags=args.lags,
        min_policy_days=args.min_policy_days,
    )
    print(f"  Total (region, policy, lag) observations: {len(lag_df)}")

    if len(lag_df) > 0:
        lag_df.to_csv(output_dir / "lag_test_data.csv", index=False)
        lag_results = analyze_lag_test(lag_df)
        if isinstance(lag_results, list) and len(lag_results) > 0:
            for lr in lag_results:
                print(f"  Lag {lr['lag']:3d}d: corr={lr['pearson_corr']:.4f}, "
                      f"RMSE={lr['rmse']:.4f}, MAE={lr['mae']:.4f}, n={lr['n_pairs']}")
            plot_lag_degradation(lag_results, output_dir)

            corr_0 = next((lr["pearson_corr"] for lr in lag_results if lr["lag"] == 0), None)
            corr_max_lag = lag_results[-1]["pearson_corr"] if lag_results else None
            if corr_0 is not None and corr_max_lag is not None:
                degradation = corr_0 - corr_max_lag
                lag_interpretation = ""
                if degradation > 0.15:
                    lag_interpretation = f"SUBSTANTIAL degradation ({degradation:.3f}) in gamma correlation with lag: supports causal NPI timing signal."
                elif degradation > 0.05:
                    lag_interpretation = f"MODERATE degradation ({degradation:.3f}): some NPI timing signal detected but gamma also captures background trends."
                else:
                    lag_interpretation = f"MINIMAL degradation ({degradation:.3f}): gamma estimates may be driven partly by smooth background trends rather than NPI-specific timing."
                print(f"\n  >>> {lag_interpretation}")
                analysis["lag_degradation"] = degradation
                analysis["lag_interpretation"] = lag_interpretation
        else:
            lag_results = []
    else:
        lag_results = []

    # --- Summary ---
    summary = {
        "split_half": analysis,
        "lag_falsification": lag_results if isinstance(lag_results, list) else [],
        "config": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "min_days_per_half": args.min_days_per_half,
            "min_policy_days": args.min_policy_days,
            "lags": args.lags,
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  All outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
