"""
Compare THEMIS with rank-1 (k_R * gamma_i) vs rank-3 gamma estimation.

Produces:
  1. Side-by-side gamma tables for 6 focus regions (DE, ES, BR, US-NY, FR, US-FL)
  2. Scatter plots: rank-1 vs rank-3 gammas across all 210 regions
  3. DELPHI simulations with both gamma sets for ALL eligible regions
  4. Publication-quality summary table and plots
"""
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from analyze_gamma_rank import (
    PARAM_COLS,
    POLICY_INDEX,
    POLICY_NAMES,
    N_POLICIES,
    build_gamma_matrix,
    rank1_imputation,
)
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
from run_themis_region_holdout import (
    _active_policy_from_row,
    _build_historical_policy_vector,
    _build_totalcases,
    _compute_actual_counts,
    _compute_gamma_stats,
    _load_global_true_data,
    _read_policy_data,
    _region_id,
    _register_region,
    prepare_global_artifacts,
)


PAPER_REGIONS = {
    "Germany__None": ("Germany", "None"),
    "Spain__None": ("Spain", "None"),
    "Brazil__None": ("Brazil", "None"),
    "US__New_York": ("US", "New York"),
}

PAPER_REGION_DISPLAY = {
    "Germany__None": "Germany",
    "Spain__None": "Spain",
    "Brazil__None": "Brazil",
    "US__New_York": "New York (US)",
}

SHORT_POLICY_NAMES = {
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others": "Auth.Schools+Restrict Others",
    "Lockdown": "Lockdown",
    "Mass_Gatherings_Authorized_But_Others_Restricted": "Gatherings Auth.+Others Restr.",
    "No_Measure": "No Measure",
    "Restrict_Mass_Gatherings": "Restrict Gatherings",
    "Restrict_Mass_Gatherings_and_Schools": "Restrict Gatherings+Schools",
    "Restrict_Mass_Gatherings_and_Schools_and_Others": "Restrict All",
}

POLICY_SEVERITY_ORDER = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Mass_Gatherings_Authorized_But_Others_Restricted",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]

POLICY_SEVERITY_LABELS = [
    "No\nMeasure",
    "Restrict\nGatherings",
    "Restrict\nGath.+Others",
    "Restrict\nGath.+Others\n(not Schools)",
    "Restrict\nGath.+Schools",
    "Restrict\nAll",
    "Lockdown",
]

OUTPUT_DIR = Path("simulation_results/rank1_vs_rank3")


# ---------------------------------------------------------------------------
# Rank-3 imputation via truncated SVD with iterative EM
# ---------------------------------------------------------------------------

def rank_r_imputation(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    rank: int = 3,
    max_iter: int = 500,
    tol: float = 1e-9,
    warm_start: np.ndarray = None,
    perturb_scale: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Complete the gamma matrix using rank-r truncated SVD (EM-style),
    with non-negative projection applied at each iteration.

    Initialization: if ``warm_start`` is provided, missing entries are
    taken from it; otherwise they are filled with the rank-1 solution.
    Both rank-2 and rank-3 should be warm-started from the rank-1
    solution so each independently explores the additional latent
    factors warranted by the data.

    When ``perturb_scale > 0``, Gaussian noise proportional to
    each column's standard deviation is added to missing entries
    before iterating.  This breaks symmetry so that higher-rank
    models do not collapse to a lower-rank fixed point (e.g. rank-3
    collapsing to rank-1).

    Iterate: clamp non-negative -> restore observed -> SVD -> truncate
    to rank r -> fill missing entries -> repeat until convergence.
    """
    rng = np.random.default_rng(seed)
    if warm_start is not None:
        X = warm_start.copy()
    else:
        rank1_filled, _ = rank1_imputation(gamma_matrix, obs_mask)
        X = rank1_filled.copy()
    X[obs_mask] = gamma_matrix[obs_mask]

    if perturb_scale > 0:
        for j in range(gamma_matrix.shape[1]):
            miss_j = ~obs_mask[:, j]
            if miss_j.any():
                std_j = X[miss_j, j].std() if miss_j.sum() > 1 else 0.1
                X[miss_j, j] += rng.normal(0, perturb_scale * std_j,
                                           size=miss_j.sum())
                np.clip(X[miss_j, j], 0, None, out=X[miss_j, j])

    for it in range(max_iter):
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

    col_mean = X.mean(axis=0, keepdims=True)
    X_c = X - col_mean
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    S[rank:] = 0.0
    X_final = col_mean + U @ np.diag(S) @ Vt

    np.clip(X_final, 0.0, None, out=X_final)
    X_final[obs_mask] = gamma_matrix[obs_mask]

    return X_final


# ---------------------------------------------------------------------------
# Build gamma dicts for DELPHI from a completed matrix
# ---------------------------------------------------------------------------

def matrix_row_to_gamma_dict(completed_row: np.ndarray) -> Dict[str, float]:
    return {p: float(completed_row[POLICY_INDEX[p]]) for p in POLICY_NAMES}


# ---------------------------------------------------------------------------
# Prepare artifacts and run DELPHI for a given region
# ---------------------------------------------------------------------------

def run_delphi_for_region(
    region_id: str,
    gamma_dict: Dict[str, float],
    artifacts: Dict[str, dict],
    policy_vector_override: Optional[List[str]] = None,
) -> Optional[Tuple[float, float]]:
    """Run DELPHI simulation for a region with a given gamma dict.
    Returns (predicted_cases, predicted_deaths) or None on failure."""
    if region_id not in artifacts:
        return None
    art = artifacts[region_id]
    _register_region(region_id, art["country"], art["province"], art["continent"])

    pv = policy_vector_override if policy_vector_override else art["policy_vector"]
    policy = Policy(
        policy_type="hypothetical",
        start_date=art["start_date_used"],
        policy_vector=pv,
    )
    try:
        out = run_delphi_policy_scenario(policy, region_id, art["totalcases"], gamma_dict)
        return float(out[0]), float(out[3])
    except Exception as e:
        print(f"  DELPHI failed for {region_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_gamma_scatter(
    rank1_gammas: np.ndarray,
    rank3_gammas: np.ndarray,
    obs_mask: np.ndarray,
    output_dir: Path,
) -> None:
    """Scatter of rank-1 vs rank-3 gamma values for all region-policy pairs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    obs_r1 = rank1_gammas[obs_mask]
    obs_r3 = rank3_gammas[obs_mask]
    unobs_r1 = rank1_gammas[~obs_mask]
    unobs_r3 = rank3_gammas[~obs_mask]

    ax1.scatter(obs_r1, obs_r3, alpha=0.4, s=18, c="#2c7fb8", label="Observed entries")
    lo = min(obs_r1.min(), obs_r3.min()) - 0.05
    hi = max(obs_r1.max(), obs_r3.max()) + 0.05
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="$y = x$")
    ax1.set_xlabel("Rank-1 $\\gamma_{R,i}$", fontsize=12)
    ax1.set_ylabel("Rank-3 $\\gamma_{R,i}$", fontsize=12)
    ax1.set_title("Observed Entries", fontsize=13)
    ax1.legend(fontsize=10)

    corr_obs = np.corrcoef(obs_r1, obs_r3)[0, 1]
    rmse_obs = np.sqrt(np.mean((obs_r1 - obs_r3) ** 2))
    ax1.text(0.05, 0.92, f"$r = {corr_obs:.4f}$\nRMSE $= {rmse_obs:.4f}$",
             transform=ax1.transAxes, fontsize=11, verticalalignment="top",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    ax2.scatter(unobs_r1, unobs_r3, alpha=0.3, s=18, c="#d95f02", label="Imputed entries")
    lo2 = min(unobs_r1.min(), unobs_r3.min()) - 0.05
    hi2 = max(unobs_r1.max(), unobs_r3.max()) + 0.05
    ax2.plot([lo2, hi2], [lo2, hi2], "k--", lw=1, alpha=0.5, label="$y = x$")
    ax2.set_xlabel("Rank-1 $\\gamma_{R,i}$", fontsize=12)
    ax2.set_ylabel("Rank-3 $\\gamma_{R,i}$", fontsize=12)
    ax2.set_title("Imputed (Unobserved) Entries", fontsize=13)
    ax2.legend(fontsize=10)

    corr_unobs = np.corrcoef(unobs_r1, unobs_r3)[0, 1]
    rmse_unobs = np.sqrt(np.mean((unobs_r1 - unobs_r3) ** 2))
    ax2.text(0.05, 0.92, f"$r = {corr_unobs:.4f}$\nRMSE $= {rmse_unobs:.4f}$",
             transform=ax2.transAxes, fontsize=11, verticalalignment="top",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    plt.tight_layout()
    fig.savefig(output_dir / "gamma_scatter_rank1_vs_rank3.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "gamma_scatter_rank1_vs_rank3.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_paper_region_gammas(
    rank1_gammas: np.ndarray,
    rank2_gammas: np.ndarray,
    rank3_gammas: np.ndarray,
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    region_ids: List[str],
    output_dir: Path,
) -> None:
    """Bar chart comparing rank-1, rank-2, and rank-3 gammas for paper regions.

    - 2 rows x 2 columns layout (4 regions)
    - Policies in increasing severity order (matching the paper)
    - Rank-1, rank-2, and rank-3 bars side by side
    - Observed policies marked with asterisk (*)
    """
    paper_rids = [r for r in PAPER_REGIONS if r in region_ids]
    if not paper_rids:
        print("  No paper regions found in data; skipping bar chart.")
        return

    severity_idx = [POLICY_INDEX[p] for p in POLICY_SEVERITY_ORDER]

    n_regions = len(paper_rids)
    n_cols = 2
    n_rows = (n_regions + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows),
                             sharey=False)
    axes_flat = axes.flatten() if n_regions > 1 else [axes]

    x = np.arange(len(POLICY_SEVERITY_ORDER))
    w = 0.25

    for idx, rid in enumerate(paper_rids):
        ax = axes_flat[idx]
        ri = region_ids.index(rid)

        r1_vals = rank1_gammas[ri][severity_idx]
        r2_vals = rank2_gammas[ri][severity_idx]
        r3_vals = rank3_gammas[ri][severity_idx]
        obs_flags = obs_mask[ri][severity_idx]

        ax.bar(x - w, r1_vals, w, label="Rank-1", color="#7570b3", alpha=0.85)
        ax.bar(x,     r2_vals, w, label="Rank-2", color="#1b9e77", alpha=0.85)
        ax.bar(x + w, r3_vals, w, label="Rank-3", color="#d95f02", alpha=0.85)

        for i in range(len(POLICY_SEVERITY_ORDER)):
            if obs_flags[i]:
                top = max(r1_vals[i], r2_vals[i], r3_vals[i])
                ax.text(x[i], top + 0.02, "*", ha="center", va="bottom",
                        fontsize=14, fontweight="bold", color="black")

        ax.set_xticks(x)
        ax.set_xticklabels(POLICY_SEVERITY_LABELS, fontsize=7.5)
        ax.set_title(PAPER_REGION_DISPLAY.get(rid, rid), fontsize=14,
                     fontweight="bold")
        ax.set_ylabel("$\\gamma_{R,i}$", fontsize=12)
        ax.legend(fontsize=9, loc="best")
        ax.axhline(y=0, color="gray", lw=0.5, ls="--", alpha=0.4)

    for idx in range(len(paper_rids), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.text(0.5, -0.01,
             "* = policy observed in data for this region",
             ha="center", fontsize=10, fontstyle="italic")

    plt.tight_layout()
    fig.savefig(output_dir / "paper_regions_gamma_comparison.pdf",
                bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "paper_regions_gamma_comparison.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_delphi_comparison(delphi_df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart comparing rank-1 and rank-3 DELPHI predicted cases and deaths."""
    paper_rows = delphi_df[delphi_df["region_id"].isin(PAPER_REGIONS)].copy()
    if paper_rows.empty:
        print("  No paper regions in DELPHI results; skipping bar chart.")
        return

    paper_rows["display_name"] = paper_rows["region_id"].map(PAPER_REGION_DISPLAY)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    regions = paper_rows["display_name"].tolist()
    x = np.arange(len(regions))
    w = 0.25

    ax1.bar(x - w, paper_rows["actual_cases"], w, label="Actual", color="#1b9e77", alpha=0.85)
    ax1.bar(x, paper_rows["rank1_cases"], w, label="Rank-1", color="#7570b3", alpha=0.85)
    ax1.bar(x + w, paper_rows["rank3_cases"], w, label="Rank-3", color="#d95f02", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, fontsize=11)
    ax1.set_ylabel("Total Detected Cases", fontsize=12)
    ax1.set_title("Predicted Cases (3-month window)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax2.bar(x - w, paper_rows["actual_deaths"], w, label="Actual", color="#1b9e77", alpha=0.85)
    ax2.bar(x, paper_rows["rank1_deaths"], w, label="Rank-1", color="#7570b3", alpha=0.85)
    ax2.bar(x + w, paper_rows["rank3_deaths"], w, label="Rank-3", color="#d95f02", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, fontsize=11)
    ax2.set_ylabel("Total Detected Deaths", fontsize=12)
    ax2.set_title("Predicted Deaths (3-month window)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    fig.savefig(output_dir / "delphi_cases_deaths_comparison.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "delphi_cases_deaths_comparison.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_holdout_gamma(ho_df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: true gamma vs predicted gamma for rank-1 and rank-3 holdout."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    ax1.scatter(ho_df["true_gamma"], ho_df["rank1_pred"], alpha=0.3, s=14, c="#7570b3")
    lo = min(ho_df["true_gamma"].min(), ho_df["rank1_pred"].min()) - 0.05
    hi = max(ho_df["true_gamma"].max(), ho_df["rank1_pred"].max()) + 0.05
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("True $\\gamma_{R,i}$", fontsize=12)
    ax1.set_ylabel("Predicted $\\gamma_{R,i}$", fontsize=12)
    ax1.set_title("Rank-1 Holdout Predictions", fontsize=13)
    rmse1 = np.sqrt((ho_df["rank1_sq_error"]).mean())
    ax1.text(0.05, 0.92, f"RMSE = {rmse1:.4f}",
             transform=ax1.transAxes, fontsize=11, verticalalignment="top",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    ax2.scatter(ho_df["true_gamma"], ho_df["rank3_pred"], alpha=0.3, s=14, c="#d95f02")
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    ax2.set_xlabel("True $\\gamma_{R,i}$", fontsize=12)
    ax2.set_ylabel("Predicted $\\gamma_{R,i}$", fontsize=12)
    ax2.set_title("Rank-3 Holdout Predictions", fontsize=13)
    rmse3 = np.sqrt((ho_df["rank3_sq_error"]).mean())
    ax2.text(0.05, 0.92, f"RMSE = {rmse3:.4f}",
             transform=ax2.transAxes, fontsize=11, verticalalignment="top",
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    plt.tight_layout()
    fig.savefig(output_dir / "holdout_scatter_rank1_vs_rank3.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "holdout_scatter_rank1_vs_rank3.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Holdout comparison: rank-1 vs rank-3
# ---------------------------------------------------------------------------

def holdout_gamma_comparison(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    n_splits: int = 20,
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    obs_indices = np.argwhere(obs_mask)
    n_obs = len(obs_indices)
    n_holdout = max(1, int(n_obs * holdout_frac))

    rows = []
    for split in range(n_splits):
        perm = rng.permutation(n_obs)
        ho_idx = obs_indices[perm[:n_holdout]]
        train_mask = obs_mask.copy()
        for r, c in ho_idx:
            train_mask[r, c] = False

        r1, _ = rank1_imputation(gamma_matrix, train_mask)
        r3 = rank_r_imputation(gamma_matrix, train_mask, rank=3)

        for r, c in ho_idx:
            tv = gamma_matrix[r, c]
            rows.append({
                "split": split,
                "region_idx": r,
                "policy_idx": c,
                "policy_name": POLICY_NAMES[c],
                "true_gamma": tv,
                "rank1_pred": r1[r, c],
                "rank3_pred": r3[r, c],
                "rank1_error": abs(r1[r, c] - tv),
                "rank3_error": abs(r3[r, c] - tv),
                "rank1_sq_error": (r1[r, c] - tv) ** 2,
                "rank3_sq_error": (r3[r, c] - tv) ** 2,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  RANK-1 vs RANK-3: COMPARISON OF GAMMA ESTIMATES AND PREDICTIONS")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Build the gamma matrix and compute both imputations
    # ------------------------------------------------------------------
    print("\n[1] Building gamma matrix ...")
    gamma_matrix, obs_mask, region_ids, policy_names = build_gamma_matrix(
        start_date="2020-03-15", end_date="2020-06-15", min_policy_days=10
    )
    n_regions, n_policies = gamma_matrix.shape
    print(f"    {n_regions} regions x {n_policies} policies, "
          f"{obs_mask.sum()} observed ({100*obs_mask.sum()/(n_regions*n_policies):.1f}%)")

    print("\n[2] Computing rank-1 imputation (current THEMIS) ...")
    rank1_completed, k_R = rank1_imputation(gamma_matrix, obs_mask)

    print("[3] Computing rank-2 imputation (warm-started from rank-1) ...")
    rank2_completed = rank_r_imputation(gamma_matrix, obs_mask, rank=2,
                                         warm_start=rank1_completed)

    print("[4] Computing rank-3 imputation (warm-started from rank-1, "
          "perturbed) ...")
    rank3_completed = rank_r_imputation(gamma_matrix, obs_mask, rank=3,
                                         warm_start=rank1_completed,
                                         perturb_scale=0.1,
                                         max_iter=2000)

    # ------------------------------------------------------------------
    # 2. Global agreement statistics
    # ------------------------------------------------------------------
    print("\n[5] Agreement between ranks ...")

    for label, other in [("rank-2", rank2_completed), ("rank-3", rank3_completed)]:
        obs_r1 = rank1_completed[obs_mask]
        obs_other = other[obs_mask]
        all_r1 = rank1_completed.ravel()
        all_other = other.ravel()
        corr_obs = np.corrcoef(obs_r1, obs_other)[0, 1]
        rmse_obs = np.sqrt(np.mean((obs_r1 - obs_other) ** 2))
        corr_all = np.corrcoef(all_r1, all_other)[0, 1]
        max_diff = np.max(np.abs(all_r1 - all_other))
        print(f"    R1 vs {label}:  obs corr={corr_obs:.4f}, obs RMSE={rmse_obs:.4f}, "
              f"all corr={corr_all:.4f}, max|diff|={max_diff:.4f}")

    plot_gamma_scatter(rank1_completed, rank3_completed, obs_mask, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 3. Paper region gamma tables
    # ------------------------------------------------------------------
    print("\n[5] Gamma tables for paper regions ...")

    paper_gamma_rows = []
    for rid in PAPER_REGIONS:
        if rid not in region_ids:
            print(f"    {rid} not in data, skipping")
            continue
        ri = region_ids.index(rid)
        for pi, pname in enumerate(POLICY_NAMES):
            obs_val = gamma_matrix[ri, pi] if obs_mask[ri, pi] else np.nan
            paper_gamma_rows.append({
                "Region": PAPER_REGION_DISPLAY.get(rid, rid),
                "Policy": SHORT_POLICY_NAMES[pname],
                "Observed": f"{obs_val:.4f}" if np.isfinite(obs_val) else "--",
                "Rank-1": f"{rank1_completed[ri, pi]:.4f}",
                "Rank-2": f"{rank2_completed[ri, pi]:.4f}",
                "Rank-3": f"{rank3_completed[ri, pi]:.4f}",
            })

    paper_gamma_df = pd.DataFrame(paper_gamma_rows)
    paper_gamma_df.to_csv(OUTPUT_DIR / "paper_regions_gamma_table.csv", index=False)
    print(f"\n{paper_gamma_df.to_string(index=False)}")

    plot_paper_region_gammas(
        rank1_completed, rank2_completed, rank3_completed, gamma_matrix,
        obs_mask, region_ids, OUTPUT_DIR,
    )

    # ------------------------------------------------------------------
    # 4. Holdout comparison: rank-1 vs rank-3
    # ------------------------------------------------------------------
    print("\n[6] Holdout comparison (20 splits, 20% holdout) ...")
    ho_df = holdout_gamma_comparison(gamma_matrix, obs_mask, n_splits=20,
                                     holdout_frac=0.2, seed=42)

    agg = ho_df.agg({
        "rank1_sq_error": "mean",
        "rank3_sq_error": "mean",
        "rank1_error": "mean",
        "rank3_error": "mean",
    })
    r1_rmse = np.sqrt(agg["rank1_sq_error"])
    r3_rmse = np.sqrt(agg["rank3_sq_error"])
    print(f"    Rank-1 holdout RMSE: {r1_rmse:.4f}   MAE: {agg['rank1_error']:.4f}")
    print(f"    Rank-3 holdout RMSE: {r3_rmse:.4f}   MAE: {agg['rank3_error']:.4f}")
    print(f"    RMSE improvement:    {100*(1 - r3_rmse/r1_rmse):.1f}%")

    per_policy = ho_df.groupby("policy_name").agg(
        rank1_rmse=("rank1_sq_error", lambda x: np.sqrt(x.mean())),
        rank3_rmse=("rank3_sq_error", lambda x: np.sqrt(x.mean())),
        rank1_mae=("rank1_error", "mean"),
        rank3_mae=("rank3_error", "mean"),
        n=("true_gamma", "count"),
    ).reset_index()
    per_policy["policy_short"] = per_policy["policy_name"].map(SHORT_POLICY_NAMES)
    print(f"\n    Per-policy holdout RMSE:\n{per_policy.to_string(index=False)}")

    ho_df.to_csv(OUTPUT_DIR / "holdout_detail.csv", index=False)
    per_policy.to_csv(OUTPUT_DIR / "holdout_per_policy.csv", index=False)
    plot_holdout_gamma(ho_df, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 5. DELPHI downstream predictions: rank-1 vs rank-3
    #    Run ALL eligible regions on their historical policy, then run
    #    paper regions on hypothetical scenarios.
    # ------------------------------------------------------------------
    print("\n[7] Running DELPHI simulations for ALL eligible regions ...")
    start_dt = datetime(2020, 3, 15)
    months = 3
    artifacts = prepare_global_artifacts(
        start_date=start_dt, months=months, min_true_days=30, max_regions=0,
    )
    print(f"    Eligible regions with truth data: {len(artifacts)}")

    region_id_to_idx = {r: i for i, r in enumerate(region_ids)}

    # --- 5a. Historical policy for ALL regions ---
    hist_rows = []
    n_run = 0
    for rid in sorted(artifacts.keys()):
        ri_idx = region_id_to_idx.get(rid)
        if ri_idx is None:
            continue

        r1_dict = matrix_row_to_gamma_dict(rank1_completed[ri_idx])
        r3_dict = matrix_row_to_gamma_dict(rank3_completed[ri_idx])

        r1_result = run_delphi_for_region(rid, r1_dict, artifacts)
        r3_result = run_delphi_for_region(rid, r3_dict, artifacts)
        if r1_result is None or r3_result is None:
            continue

        art = artifacts[rid]
        hist_rows.append({
            "region_id": rid,
            "display_name": PAPER_REGION_DISPLAY.get(rid, rid.replace("__", " / ")),
            "actual_cases": art["actual_cases"],
            "actual_deaths": art["actual_deaths"],
            "rank1_cases": r1_result[0],
            "rank1_deaths": r1_result[1],
            "rank3_cases": r3_result[0],
            "rank3_deaths": r3_result[1],
        })
        n_run += 1

    hist_df = pd.DataFrame(hist_rows)
    if not hist_df.empty:
        hist_df["cases_diff_pct"] = 100 * (
            hist_df["rank3_cases"] - hist_df["rank1_cases"]
        ).abs() / (hist_df["rank1_cases"].abs() + 1e-6)
        hist_df["deaths_diff_pct"] = 100 * (
            hist_df["rank3_deaths"] - hist_df["rank1_deaths"]
        ).abs() / (hist_df["rank1_deaths"].abs() + 1e-6)

        hist_df.to_csv(OUTPUT_DIR / "delphi_historical_all_regions.csv", index=False)

        corr_cases_hist = np.corrcoef(hist_df["rank1_cases"], hist_df["rank3_cases"])[0, 1]
        corr_deaths_hist = np.corrcoef(hist_df["rank1_deaths"], hist_df["rank3_deaths"])[0, 1]

        print(f"\n    Historical policy DELPHI results ({len(hist_df)} regions):")
        print(f"    Cases  correlation (R1 vs R3): {corr_cases_hist:.4f}")
        print(f"    Deaths correlation (R1 vs R3): {corr_deaths_hist:.4f}")
        print(f"    Median |cases diff|:  {hist_df['cases_diff_pct'].median():.2f}%")
        print(f"    Median |deaths diff|: {hist_df['deaths_diff_pct'].median():.2f}%")
        print(f"    Mean   |cases diff|:  {hist_df['cases_diff_pct'].mean():.2f}%")
        print(f"    Mean   |deaths diff|: {hist_df['deaths_diff_pct'].mean():.2f}%")

        paper_hist = hist_df[hist_df["region_id"].isin(PAPER_REGIONS)].copy()
        if not paper_hist.empty:
            print(f"\n    Paper regions historical:")
            print(paper_hist[["display_name", "actual_cases", "rank1_cases", "rank3_cases",
                              "cases_diff_pct", "actual_deaths", "rank1_deaths",
                              "rank3_deaths", "deaths_diff_pct"]].to_string(index=False))

        # Plot: scatter rank1 vs rank3 predictions for all regions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        ax1.scatter(hist_df["rank1_cases"], hist_df["rank3_cases"],
                    alpha=0.5, s=20, c="#2c7fb8")
        lo = min(hist_df["rank1_cases"].min(), hist_df["rank3_cases"].min())
        hi = max(hist_df["rank1_cases"].max(), hist_df["rank3_cases"].max())
        ax1.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax1.set_xlabel("Rank-1 Predicted Cases", fontsize=12)
        ax1.set_ylabel("Rank-3 Predicted Cases", fontsize=12)
        ax1.set_title(f"Historical Cases ({len(hist_df)} regions)", fontsize=13)
        ax1.text(0.05, 0.92, f"$r = {corr_cases_hist:.4f}$",
                 transform=ax1.transAxes, fontsize=11, verticalalignment="top",
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

        ax2.scatter(hist_df["rank1_deaths"], hist_df["rank3_deaths"],
                    alpha=0.5, s=20, c="#d95f02")
        lo2 = min(hist_df["rank1_deaths"].min(), hist_df["rank3_deaths"].min())
        hi2 = max(hist_df["rank1_deaths"].max(), hist_df["rank3_deaths"].max())
        ax2.plot([lo2, hi2], [lo2, hi2], "k--", lw=1, alpha=0.5)
        ax2.set_xlabel("Rank-1 Predicted Deaths", fontsize=12)
        ax2.set_ylabel("Rank-3 Predicted Deaths", fontsize=12)
        ax2.set_title(f"Historical Deaths ({len(hist_df)} regions)", fontsize=13)
        ax2.text(0.05, 0.92, f"$r = {corr_deaths_hist:.4f}$",
                 transform=ax2.transAxes, fontsize=11, verticalalignment="top",
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "delphi_historical_scatter_all.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(OUTPUT_DIR / "delphi_historical_scatter_all.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        corr_cases_hist, corr_deaths_hist = np.nan, np.nan

    # --- 5b. Hypothetical scenarios for paper regions ---
    HYPOTHETICAL_SCENARIOS = [
        ["Lockdown", "Lockdown", "Lockdown"],
        ["Restrict_Mass_Gatherings_and_Schools_and_Others",
         "Restrict_Mass_Gatherings_and_Schools_and_Others",
         "Restrict_Mass_Gatherings_and_Schools_and_Others"],
        ["Lockdown",
         "Restrict_Mass_Gatherings_and_Schools_and_Others",
         "Restrict_Mass_Gatherings"],
        ["No_Measure", "Restrict_Mass_Gatherings", "Lockdown"],
        ["Lockdown", "Restrict_Mass_Gatherings_and_Schools", "No_Measure"],
    ]
    SCENARIO_LABELS = [
        "Full Lockdown",
        "Full Restrict All",
        "Lock->Restr.All->Restr.Gath.",
        "None->Restr.Gath.->Lockdown",
        "Lock->Restr.G+S->None",
    ]

    hyp_rows = []
    for rid in sorted(PAPER_REGIONS.keys()):
        ri_idx = region_id_to_idx.get(rid)
        if ri_idx is None or rid not in artifacts:
            print(f"    {rid}: not found, skipping hypothetical scenarios.")
            continue

        r1_dict = matrix_row_to_gamma_dict(rank1_completed[ri_idx])
        r3_dict = matrix_row_to_gamma_dict(rank3_completed[ri_idx])

        for scenario_pv, scenario_label in zip(HYPOTHETICAL_SCENARIOS, SCENARIO_LABELS):
            for method_name, gdict in [("rank1", r1_dict), ("rank3", r3_dict)]:
                result = run_delphi_for_region(rid, gdict, artifacts,
                                               policy_vector_override=scenario_pv)
                if result is not None:
                    hyp_rows.append({
                        "region_id": rid,
                        "display_name": PAPER_REGION_DISPLAY.get(rid, rid),
                        "scenario": scenario_label,
                        "method": method_name,
                        "pred_cases": result[0],
                        "pred_deaths": result[1],
                    })

    hyp_df = pd.DataFrame(hyp_rows)
    hyp_df.to_csv(OUTPUT_DIR / "delphi_hypothetical_paper_regions.csv", index=False)

    corr_cases = np.nan
    corr_deaths = np.nan
    med_diff_cases = np.nan
    med_diff_deaths = np.nan

    if not hyp_df.empty:
        pivot_cases = hyp_df.pivot_table(
            index=["display_name", "scenario"], columns="method", values="pred_cases",
        ).reset_index()
        pivot_deaths = hyp_df.pivot_table(
            index=["display_name", "scenario"], columns="method", values="pred_deaths",
        ).reset_index()

        if "rank1" in pivot_cases.columns and "rank3" in pivot_cases.columns:
            pivot_cases["diff_pct"] = 100 * (
                pivot_cases["rank3"] - pivot_cases["rank1"]
            ).abs() / (pivot_cases["rank1"].abs() + 1e-6)
            pivot_deaths["diff_pct"] = 100 * (
                pivot_deaths["rank3"] - pivot_deaths["rank1"]
            ).abs() / (pivot_deaths["rank1"].abs() + 1e-6)

            vc = pivot_cases.dropna(subset=["rank1", "rank3"])
            vd = pivot_deaths.dropna(subset=["rank1", "rank3"])
            corr_cases = np.corrcoef(vc["rank1"], vc["rank3"])[0, 1]
            corr_deaths = np.corrcoef(vd["rank1"], vd["rank3"])[0, 1]
            med_diff_cases = vc["diff_pct"].median()
            med_diff_deaths = vd["diff_pct"].median()

        pivot_cases.to_csv(OUTPUT_DIR / "delphi_pivot_cases.csv", index=False)
        pivot_deaths.to_csv(OUTPUT_DIR / "delphi_pivot_deaths.csv", index=False)
        print(f"\n    Hypothetical scenarios (paper regions) cases:")
        print(pivot_cases.to_string(index=False))
        print(f"\n    Hypothetical scenarios (paper regions) deaths:")
        print(pivot_deaths.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Summary JSON
    # ------------------------------------------------------------------
    obs_r1_r3 = rank1_completed[obs_mask]
    obs_r3_r3 = rank3_completed[obs_mask]
    _corr_obs = np.corrcoef(obs_r1_r3, obs_r3_r3)[0, 1]
    _rmse_obs = np.sqrt(np.mean((obs_r1_r3 - obs_r3_r3) ** 2))

    summary = {
        "n_regions": int(n_regions),
        "n_policies": int(n_policies),
        "n_observed": int(obs_mask.sum()),
        "gamma_agreement_r1_vs_r3": {
            "observed_corr": round(float(_corr_obs), 6),
            "observed_rmse": round(float(_rmse_obs), 6),
        },
        "holdout": {
            "n_splits": 20,
            "holdout_frac": 0.2,
            "rank1_rmse": round(float(r1_rmse), 6),
            "rank3_rmse": round(float(r3_rmse), 6),
            "rank1_mae": round(float(agg["rank1_error"]), 6),
            "rank3_mae": round(float(agg["rank3_error"]), 6),
            "rmse_improvement_pct": round(float(100 * (1 - r3_rmse / r1_rmse)), 2),
        },
    }
    if not hist_df.empty:
        summary["delphi_historical_all_regions"] = {
            "n_regions": len(hist_df),
            "cases_corr": round(float(corr_cases_hist), 6) if np.isfinite(corr_cases_hist) else None,
            "deaths_corr": round(float(corr_deaths_hist), 6) if np.isfinite(corr_deaths_hist) else None,
            "median_cases_diff_pct": round(float(hist_df["cases_diff_pct"].median()), 2),
            "median_deaths_diff_pct": round(float(hist_df["deaths_diff_pct"].median()), 2),
            "mean_cases_diff_pct": round(float(hist_df["cases_diff_pct"].mean()), 2),
            "mean_deaths_diff_pct": round(float(hist_df["deaths_diff_pct"].mean()), 2),
        }
    if np.isfinite(corr_cases):
        summary["delphi_hypothetical_paper_regions"] = {
            "scenarios_corr_cases": round(float(corr_cases), 6),
            "scenarios_corr_deaths": round(float(corr_deaths), 6),
            "median_cases_diff_pct": round(float(med_diff_cases), 2),
            "median_deaths_diff_pct": round(float(med_diff_deaths), 2),
        }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    print(f"  Gamma agreement R1 vs R3 (obs): corr = {_corr_obs:.4f}, RMSE = {_rmse_obs:.4f}")
    print(f"  Holdout RMSE rank-1: {r1_rmse:.4f}, rank-3: {r3_rmse:.4f} "
          f"(improvement: {summary['holdout']['rmse_improvement_pct']:.1f}%)")
    if not hist_df.empty:
        print(f"  DELPHI historical ({len(hist_df)} regions):")
        print(f"    Cases corr:  {corr_cases_hist:.4f}   Median diff: {hist_df['cases_diff_pct'].median():.1f}%")
        print(f"    Deaths corr: {corr_deaths_hist:.4f}   Median diff: {hist_df['deaths_diff_pct'].median():.1f}%")
    if np.isfinite(corr_cases):
        print(f"  DELPHI hypothetical (paper regions):")
        print(f"    Cases corr:  {corr_cases:.4f}   Deaths corr: {corr_deaths:.4f}")
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
