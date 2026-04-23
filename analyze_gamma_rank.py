"""
Low-rank analysis of the gamma_{R,i} matrix.

Validates (or refutes) the separability assumption gamma_{R,i} = k_R * gamma_i
by computing the SVD of the observed gamma matrix and comparing rank-1 vs
low-rank matrix completion for imputing unobserved entries.

Outputs:
  - SVD scree plot and explained-variance table
  - Nuclear-norm matrix completion with cross-validated lambda
  - Holdout comparison of rank-1 vs low-rank imputation
"""
import argparse
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
from scipy.optimize import minimize

from pandemic_functions.pandemic_params import (
    default_dict_normalized_policy_gamma,
    future_policies,
)
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    get_dominant_policy,
    past_parameters,
    read_oxford_country_policy_data,
    read_policy_data_us_only,
)
from pandemic_functions.delphi_functions.DELPHI_utils import gamma_t


POLICY_NAMES = sorted(default_dict_normalized_policy_gamma.keys())
N_POLICIES = len(POLICY_NAMES)
POLICY_INDEX = {p: i for i, p in enumerate(POLICY_NAMES)}

PARAM_COLS = [
    "Data Start Date",
    "Median Day of Action",
    "Rate of Action",
    "Jump Magnitude",
    "Jump Time",
    "Jump Decay",
]


# ---------------------------------------------------------------------------
# Step 1: Assemble the partially-observed gamma matrix
# ---------------------------------------------------------------------------

def _read_policy_data(country: str, province: str,
                      start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    try:
        if country == "US":
            df = read_policy_data_us_only(state=province,
                                          start_date=start_date,
                                          end_date=end_date)
        else:
            df = read_oxford_country_policy_data(country=country,
                                                 start_date=start_date,
                                                 end_date=end_date)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    return df if len(df) > 0 else None


def _active_policy_from_row(row: pd.Series) -> Optional[str]:
    for policy in future_policies:
        if row.get(policy, 0) == 1:
            return policy
    return None


def build_gamma_matrix(
    start_date: str = "2020-03-01",
    end_date: str = "2020-07-31",
    min_policy_days: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build the partially-observed gamma matrix Gamma[R, i].

    Returns
    -------
    gamma_matrix : ndarray of shape (n_regions, n_policies)
        Observed entries filled, unobserved entries = NaN.
    obs_mask : ndarray of shape (n_regions, n_policies), dtype bool
        True where gamma_matrix has an observed value.
    region_ids : list of str
        Region identifiers (Country__Province).
    policy_names : list of str
        Sorted policy names (columns).
    """
    default_gammas = dict(sorted(
        default_dict_normalized_policy_gamma.items(), key=lambda x: x[0]
    ))

    params_unique = (
        past_parameters
        .sort_values(["Country", "Province", "Data Start Date"])
        .drop_duplicates(subset=["Country", "Province"], keep="last")
        .reset_index(drop=True)
    )

    region_ids: List[str] = []
    gamma_rows: List[np.ndarray] = []
    obs_rows: List[np.ndarray] = []

    for _, prow in params_unique.iterrows():
        country = str(prow["Country"])
        province = str(prow["Province"])
        params_list = prow[PARAM_COLS]

        policy_data = _read_policy_data(country, province, start_date, end_date)
        if policy_data is None or len(policy_data) == 0:
            continue

        policy_gamma_sum = {p: 0.0 for p in POLICY_NAMES}
        policy_gamma_count = {p: 0 for p in POLICY_NAMES}

        for _, row in policy_data.iterrows():
            policy_name = _active_policy_from_row(row)
            if policy_name is None or policy_name not in default_gammas:
                continue
            g_val = float(gamma_t(row["date"], params_list))
            policy_gamma_sum[policy_name] += g_val
            policy_gamma_count[policy_name] += 1

        row_gamma = np.full(N_POLICIES, np.nan)
        row_obs = np.zeros(N_POLICIES, dtype=bool)
        has_any = False
        for p in POLICY_NAMES:
            idx = POLICY_INDEX[p]
            if policy_gamma_count[p] >= min_policy_days:
                row_gamma[idx] = policy_gamma_sum[p] / policy_gamma_count[p]
                row_obs[idx] = True
                has_any = True

        if not has_any:
            continue

        rid = f"{country}__{province}".replace(" ", "_")
        region_ids.append(rid)
        gamma_rows.append(row_gamma)
        obs_rows.append(row_obs)

    gamma_matrix = np.array(gamma_rows)
    obs_mask = np.array(obs_rows)
    return gamma_matrix, obs_mask, region_ids, POLICY_NAMES


# ---------------------------------------------------------------------------
# Step 2: SVD analysis
# ---------------------------------------------------------------------------

def svd_analysis(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    min_obs_frac: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform SVD on the subset of regions with sufficient observations.

    Strategy: iterative SVD imputation (EM-style) to handle missing entries.
    Initialize missing entries with random values drawn uniformly from the
    observed range of gamma (rank-agnostic), then iterate full-rank SVD
    reconstruction until convergence.

    Returns
    -------
    singular_values : 1-d array
    explained_variance_ratio : 1-d array (sigma_i^2 / sum sigma_j^2)
    cumulative_variance : 1-d array
    G_sub : the filled submatrix used for SVD
    """
    n_policies = gamma_matrix.shape[1]
    obs_frac = obs_mask.sum(axis=1) / n_policies
    keep = obs_frac >= min_obs_frac
    G_sub = gamma_matrix[keep].copy()
    mask_sub = obs_mask[keep].copy()

    obs_vals = G_sub[mask_sub]
    lo, hi = float(obs_vals.min()), float(obs_vals.max())
    rng = np.random.RandomState(42)
    n_missing = int((~mask_sub).sum())
    G_filled = G_sub.copy()
    G_filled[~mask_sub] = rng.uniform(lo, hi, size=n_missing)

    for _ in range(50):
        G_centered = G_filled - G_filled.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(G_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        G_approx = G_filled.mean(axis=0, keepdims=True) + U @ np.diag(S) @ Vt
        G_new = G_filled.copy()
        G_new[~mask_sub] = G_approx[~mask_sub]
        if np.linalg.norm(G_new - G_filled) / (np.linalg.norm(G_filled) + 1e-12) < 1e-6:
            G_filled = G_new
            break
        G_filled = G_new

    G_centered = G_filled - G_filled.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(G_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        U, S, Vt = np.linalg.svd(G_centered + 1e-10 * np.random.randn(*G_centered.shape),
                                  full_matrices=False)

    var = S ** 2
    total_var = var.sum()
    explained = var / total_var if total_var > 0 else var
    cumulative = np.cumsum(explained)

    return S, explained, cumulative, G_filled


# ---------------------------------------------------------------------------
# Step 3: Nuclear-norm matrix completion (proximal gradient)
# ---------------------------------------------------------------------------

def _soft_threshold_singular(X: np.ndarray, tau: float) -> np.ndarray:
    """Singular value soft-thresholding (proximal operator for nuclear norm)."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_thresh = np.maximum(S - tau, 0.0)
    return U @ np.diag(S_thresh) @ Vt


def nuclear_norm_completion(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    lam: float,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Solve:  min_{X}  (1/2) ||P_Omega(X - Y)||_F^2  +  lam * ||X||_*

    via accelerated proximal gradient (FISTA).

    Parameters
    ----------
    gamma_matrix : observed entries (NaN for missing)
    obs_mask : boolean mask
    lam : nuclear norm penalty
    """
    Y = np.where(obs_mask, gamma_matrix, 0.0)
    X = Y.copy()
    Z = X.copy()
    t_k = 1.0

    for k in range(max_iter):
        grad = np.where(obs_mask, Z - Y, 0.0)
        X_new = _soft_threshold_singular(Z - grad, lam)

        t_new = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
        Z = X_new + ((t_k - 1) / t_new) * (X_new - X)

        change = np.linalg.norm(X_new - X) / (np.linalg.norm(X) + 1e-12)
        X = X_new
        t_k = t_new
        if change < tol:
            break

    return X


def cross_validate_lambda(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    lambdas: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    K-fold cross-validation on observed entries to select lambda.

    Returns
    -------
    lambdas : array of lambda values tested
    mean_mse : array of mean MSE for each lambda
    best_lambda : the lambda with lowest mean MSE
    """
    rng = np.random.default_rng(seed)
    obs_indices = np.argwhere(obs_mask)
    rng.shuffle(obs_indices)

    folds = np.array_split(obs_indices, n_folds)
    mean_mse = np.zeros(len(lambdas))

    for li, lam in enumerate(lambdas):
        fold_mses = []
        for fi in range(n_folds):
            val_idx = folds[fi]
            train_mask = obs_mask.copy()
            for r, c in val_idx:
                train_mask[r, c] = False

            X_hat = nuclear_norm_completion(gamma_matrix, train_mask, lam)

            val_errors = []
            for r, c in val_idx:
                val_errors.append((X_hat[r, c] - gamma_matrix[r, c]) ** 2)
            fold_mses.append(np.mean(val_errors))
        mean_mse[li] = np.mean(fold_mses)

    best_lambda = lambdas[np.argmin(mean_mse)]
    return lambdas, mean_mse, best_lambda


# ---------------------------------------------------------------------------
# Step 4: Rank-1 baseline (current THEMIS k_R * gamma_i)
# ---------------------------------------------------------------------------

def rank1_imputation(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank-1 model: gamma_{R,i} = k_R * g_i.

    Both k_R and g_i are learned from observed entries via alternating
    least squares with non-negativity constraints.  g_i is initialized
    with the calibrated default gammas; for policies with no matrix
    observations, g_i retains the default.  A second run from column-mean
    initialization is also performed, and the solution with lower
    observed-entry residual is kept.

    Returns the completed matrix and the per-region k_R values.
    """
    n_regions, n_policies = gamma_matrix.shape

    default_gammas = np.array([
        default_dict_normalized_policy_gamma[p] for p in POLICY_NAMES
    ])

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
        col_means[j] = gamma_matrix[oj, j].mean() if oj.any() else default_gammas[j]
    pol_only_resid = np.sum(
        (np.tile(col_means, (n_regions, 1))[obs_mask] - gamma_matrix[obs_mask]) ** 2
    )

    comp_def, k_def, res_def = _als_run(default_gammas)
    if res_def <= pol_only_resid:
        return comp_def, k_def

    comp_cm, k_cm, _ = _als_run(col_means)
    return comp_cm, k_cm


# ---------------------------------------------------------------------------
# Step 5: Holdout comparison
# ---------------------------------------------------------------------------

def holdout_comparison(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    best_lambda: float,
    n_splits: int = 20,
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Hold out a fraction of observed entries, impute with rank-1 and low-rank,
    then compare MSE and MAE.
    """
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

        rank1_completed, _ = rank1_imputation(gamma_matrix, train_mask)
        lowrank_completed = nuclear_norm_completion(gamma_matrix, train_mask, best_lambda)

        for r, c in ho_idx:
            true_val = gamma_matrix[r, c]
            rank1_pred = rank1_completed[r, c]
            lowrank_pred = lowrank_completed[r, c]
            rows.append({
                "split": split,
                "region_idx": r,
                "policy_idx": c,
                "policy_name": POLICY_NAMES[c],
                "true_gamma": true_val,
                "rank1_pred": rank1_pred,
                "lowrank_pred": lowrank_pred,
                "rank1_error": abs(rank1_pred - true_val),
                "lowrank_error": abs(lowrank_pred - true_val),
                "rank1_sq_error": (rank1_pred - true_val) ** 2,
                "lowrank_sq_error": (lowrank_pred - true_val) ** 2,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scree(S: np.ndarray, explained: np.ndarray, cumulative: np.ndarray,
               output_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    k = len(S)
    ax1.bar(range(1, k + 1), explained * 100, color="#2c7fb8", edgecolor="white")
    ax1.set_xlabel("Singular Value Index", fontsize=12)
    ax1.set_ylabel("Explained Variance (%)", fontsize=12)
    ax1.set_title("Scree Plot: Singular Value Decomposition of $\\Gamma$", fontsize=13)
    ax1.set_xticks(range(1, k + 1))

    for i, v in enumerate(explained * 100):
        ax1.text(i + 1, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax2.plot(range(1, k + 1), cumulative * 100, "o-", color="#d95f02", linewidth=2, markersize=8)
    ax2.axhline(y=90, color="gray", linestyle="--", alpha=0.7, label="90% threshold")
    ax2.axhline(y=95, color="gray", linestyle=":", alpha=0.7, label="95% threshold")
    ax2.set_xlabel("Number of Components", fontsize=12)
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=12)
    ax2.set_title("Cumulative Explained Variance", fontsize=13)
    ax2.set_xticks(range(1, k + 1))
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "svd_scree_plot.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "svd_scree_plot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_cv_lambda(lambdas: np.ndarray, mean_mse: np.ndarray,
                   best_lambda: float, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, mean_mse, "o-", color="#1b9e77", linewidth=2, markersize=6)
    ax.axvline(x=best_lambda, color="red", linestyle="--", alpha=0.8,
               label=f"Best $\\lambda$ = {best_lambda:.4f}")
    ax.set_xlabel("$\\lambda$ (Nuclear Norm Penalty)", fontsize=12)
    ax.set_ylabel("Mean Squared Error (CV)", fontsize=12)
    ax.set_title("Cross-Validation for Nuclear Norm Regularization", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    plt.tight_layout()
    fig.savefig(output_dir / "cv_lambda.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "cv_lambda.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_holdout_comparison(ho_df: pd.DataFrame, output_dir: Path) -> None:
    summary = ho_df.groupby("split").agg(
        rank1_rmse=("rank1_sq_error", lambda x: np.sqrt(x.mean())),
        lowrank_rmse=("lowrank_sq_error", lambda x: np.sqrt(x.mean())),
        rank1_mae=("rank1_error", "mean"),
        lowrank_mae=("lowrank_error", "mean"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(summary))
    w = 0.35
    ax1.bar(x - w / 2, summary["rank1_rmse"], w, label="Rank-1 ($k_R \\gamma_i$)", color="#7570b3")
    ax1.bar(x + w / 2, summary["lowrank_rmse"], w, label="Low-Rank (Nuclear Norm)", color="#e7298a")
    ax1.set_xlabel("Holdout Split", fontsize=12)
    ax1.set_ylabel("RMSE", fontsize=12)
    ax1.set_title("Holdout RMSE: Rank-1 vs Low-Rank", fontsize=13)
    ax1.legend(fontsize=10)

    ax2.bar(x - w / 2, summary["rank1_mae"], w, label="Rank-1 ($k_R \\gamma_i$)", color="#7570b3")
    ax2.bar(x + w / 2, summary["lowrank_mae"], w, label="Low-Rank (Nuclear Norm)", color="#e7298a")
    ax2.set_xlabel("Holdout Split", fontsize=12)
    ax2.set_ylabel("MAE", fontsize=12)
    ax2.set_title("Holdout MAE: Rank-1 vs Low-Rank", fontsize=13)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "holdout_comparison.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "holdout_comparison.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_gamma_heatmap(gamma_matrix: np.ndarray, obs_mask: np.ndarray,
                       region_ids: List[str], output_dir: Path) -> None:
    """Heatmap of the observed gamma matrix (NaN entries shown as white)."""
    short_policies = [
        "Auth.Schools\n+Restrict Others",
        "Lockdown",
        "Restrict\nGatherings+Others",
        "No Measure",
        "Restrict\nGatherings",
        "Restrict\nGatherings+Schools",
        "Restrict\nAll",
    ]

    display_matrix = gamma_matrix.copy()
    display_matrix[~obs_mask] = np.nan

    n_regions = len(region_ids)
    fig_height = max(6, n_regions * 0.12)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(display_matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(N_POLICIES))
    ax.set_xticklabels(short_policies, fontsize=8, rotation=45, ha="right")

    if n_regions <= 50:
        ax.set_yticks(range(n_regions))
        ax.set_yticklabels([r.replace("__", " / ") for r in region_ids], fontsize=6)
    else:
        ax.set_yticks([])

    ax.set_title("Observed $\\gamma_{R,i}$ Matrix", fontsize=14)
    plt.colorbar(im, ax=ax, label="$\\gamma_{R,i}$", shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / "gamma_heatmap.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-rank analysis of gamma_{R,i}")
    parser.add_argument("--start-date", default="2020-03-01")
    parser.add_argument("--end-date", default="2020-07-31")
    parser.add_argument("--min-policy-days", type=int, default=10)
    parser.add_argument("--min-obs-frac", type=float, default=0.5,
                        help="Min fraction of observed policies to include region in SVD")
    parser.add_argument("--n-holdout-splits", type=int, default=20)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="simulation_results/gamma_rank_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  LOW-RANK ANALYSIS OF THE GAMMA_{R,i} MATRIX")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Build gamma matrix
    # ------------------------------------------------------------------
    print("\n[Step 1] Building observed gamma matrix ...")
    gamma_matrix, obs_mask, region_ids, policy_names = build_gamma_matrix(
        start_date=args.start_date,
        end_date=args.end_date,
        min_policy_days=args.min_policy_days,
    )
    n_regions, n_policies = gamma_matrix.shape
    n_obs = obs_mask.sum()
    n_total = n_regions * n_policies
    print(f"  Regions: {n_regions}")
    print(f"  Policies: {n_policies}")
    print(f"  Observed entries: {n_obs} / {n_total} ({100 * n_obs / n_total:.1f}%)")
    print(f"  Policy names: {policy_names}")

    obs_per_region = obs_mask.sum(axis=1)
    print(f"  Obs per region: min={obs_per_region.min()}, "
          f"max={obs_per_region.max()}, mean={obs_per_region.mean():.1f}")

    plot_gamma_heatmap(gamma_matrix, obs_mask, region_ids, output_dir)

    # ------------------------------------------------------------------
    # Step 2: SVD analysis
    # ------------------------------------------------------------------
    print("\n[Step 2] SVD analysis ...")
    S, explained, cumulative, G_sub = svd_analysis(
        gamma_matrix, obs_mask, min_obs_frac=args.min_obs_frac,
    )
    n_svd_regions = G_sub.shape[0]
    print(f"  Regions used for SVD (>={100*args.min_obs_frac:.0f}% obs): {n_svd_regions}")
    print(f"  Singular values: {np.array2string(S, precision=4)}")
    print(f"  Explained variance ratios: {np.array2string(explained, precision=4)}")
    print(f"  Cumulative variance: {np.array2string(cumulative, precision=4)}")
    print(f"  Rank-1 explains: {100 * explained[0]:.2f}% of variance")
    if len(explained) > 1:
        print(f"  Rank-2 explains: {100 * cumulative[1]:.2f}% (cumulative)")
    if len(explained) > 2:
        print(f"  Rank-3 explains: {100 * cumulative[2]:.2f}% (cumulative)")

    plot_scree(S, explained, cumulative, output_dir)

    svd_table = pd.DataFrame({
        "Component": list(range(1, len(S) + 1)),
        "Singular Value": S,
        "Explained Variance (%)": explained * 100,
        "Cumulative Variance (%)": cumulative * 100,
    })
    svd_table.to_csv(output_dir / "svd_explained_variance.csv", index=False)
    print(f"\n  SVD table:\n{svd_table.to_string(index=False)}")

    # ------------------------------------------------------------------
    # Step 3: Nuclear-norm matrix completion with CV
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Step 2b: Direct rank-1 goodness-of-fit on observed entries only
    # ------------------------------------------------------------------
    print("\n[Step 2b] Direct rank-1 fit on observed entries ...")
    rank1_init, k_R_init = rank1_imputation(gamma_matrix, obs_mask)
    obs_residuals = np.where(obs_mask, (gamma_matrix - rank1_init) ** 2, 0.0)
    obs_frobenius = np.where(obs_mask, gamma_matrix ** 2, 0.0)
    rank1_r_squared = 1.0 - obs_residuals.sum() / (obs_frobenius.sum() + 1e-12)

    centered_gamma = gamma_matrix.copy()
    col_means_for_r2 = np.zeros(N_POLICIES)
    for j in range(N_POLICIES):
        vals = gamma_matrix[obs_mask[:, j], j]
        col_means_for_r2[j] = vals.mean() if len(vals) > 0 else 0.0
        centered_gamma[:, j] -= col_means_for_r2[j]
    total_ss = np.where(obs_mask, centered_gamma ** 2, 0.0).sum()
    residual_ss = obs_residuals.sum()
    rank1_r_squared_centered = 1.0 - residual_ss / (total_ss + 1e-12)

    print(f"  Rank-1 R^2 (uncentered, ||Y - k*g||^2 / ||Y||^2): {rank1_r_squared:.4f}")
    print(f"  Rank-1 R^2 (centered, 1 - RSS/TSS): {rank1_r_squared_centered:.4f}")
    print(f"  Rank-1 fit RMSE on observed entries: {np.sqrt(obs_residuals.sum() / obs_mask.sum()):.4f}")

    # ------------------------------------------------------------------
    # Step 3: Nuclear-norm matrix completion with CV
    # ------------------------------------------------------------------
    print("\n[Step 3] Nuclear-norm matrix completion (cross-validating lambda) ...")
    S_max = np.linalg.svd(np.where(obs_mask, gamma_matrix, 0.0), compute_uv=False)[0]
    lambdas = np.logspace(
        np.log10(max(S_max * 1e-5, 1e-6)), np.log10(S_max * 5), 20
    )

    lambdas_cv, mean_mse_cv, best_lambda = cross_validate_lambda(
        gamma_matrix, obs_mask, lambdas,
        n_folds=args.cv_folds, seed=args.seed,
    )
    print(f"  Lambda range: [{lambdas[0]:.6f}, {lambdas[-1]:.4f}]")
    print(f"  Best lambda: {best_lambda:.6f}")
    print(f"  CV MSE at best lambda: {mean_mse_cv[np.argmin(mean_mse_cv)]:.6f}")

    plot_cv_lambda(lambdas_cv, mean_mse_cv, best_lambda, output_dir)

    X_lowrank = nuclear_norm_completion(gamma_matrix, obs_mask, best_lambda)
    _, S_completed, _ = np.linalg.svd(X_lowrank - X_lowrank.mean(axis=0, keepdims=True),
                                       full_matrices=False)
    effective_rank = np.sum(S_completed > 1e-6 * S_completed[0])
    s_var_completed = S_completed ** 2
    s_total_completed = s_var_completed.sum()
    s_explained_completed = s_var_completed / s_total_completed if s_total_completed > 0 else s_var_completed
    print(f"  Effective rank of completed matrix: {effective_rank}")
    print(f"  Completed matrix singular values: {np.array2string(S_completed, precision=4)}")
    print(f"  Completed matrix explained variance: {np.array2string(s_explained_completed, precision=4)}")

    # ------------------------------------------------------------------
    # Step 4: Holdout comparison
    # ------------------------------------------------------------------
    print("\n[Step 4] Holdout comparison: rank-1 vs low-rank ...")
    ho_df = holdout_comparison(
        gamma_matrix, obs_mask, best_lambda,
        n_splits=args.n_holdout_splits,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
    )

    agg = ho_df.agg({
        "rank1_sq_error": "mean",
        "lowrank_sq_error": "mean",
        "rank1_error": "mean",
        "lowrank_error": "mean",
    })
    print(f"  Overall Rank-1 RMSE:   {np.sqrt(agg['rank1_sq_error']):.6f}")
    print(f"  Overall Low-Rank RMSE: {np.sqrt(agg['lowrank_sq_error']):.6f}")
    print(f"  Overall Rank-1 MAE:    {agg['rank1_error']:.6f}")
    print(f"  Overall Low-Rank MAE:  {agg['lowrank_error']:.6f}")

    improvement_rmse = 1 - np.sqrt(agg["lowrank_sq_error"]) / np.sqrt(agg["rank1_sq_error"])
    print(f"  Low-rank RMSE improvement over rank-1: {100 * improvement_rmse:.1f}%")

    per_policy = ho_df.groupby("policy_name").agg(
        rank1_rmse=("rank1_sq_error", lambda x: np.sqrt(x.mean())),
        lowrank_rmse=("lowrank_sq_error", lambda x: np.sqrt(x.mean())),
        rank1_mae=("rank1_error", "mean"),
        lowrank_mae=("lowrank_error", "mean"),
        n_samples=("true_gamma", "count"),
    ).reset_index()
    print(f"\n  Per-policy RMSE:\n{per_policy.to_string(index=False)}")

    ho_df.to_csv(output_dir / "holdout_detail.csv", index=False)
    per_policy.to_csv(output_dir / "holdout_per_policy.csv", index=False)
    plot_holdout_comparison(ho_df, output_dir)

    # ------------------------------------------------------------------
    # Step 5: Summary report
    # ------------------------------------------------------------------
    rank1_completed, k_R = rank1_imputation(gamma_matrix, obs_mask)
    rank1_residuals = np.where(obs_mask,
                               (gamma_matrix - rank1_completed) ** 2, 0.0)
    lowrank_residuals = np.where(obs_mask,
                                  (gamma_matrix - X_lowrank) ** 2, 0.0)

    rank1_fit_mse = rank1_residuals.sum() / obs_mask.sum()
    lowrank_fit_mse = lowrank_residuals.sum() / obs_mask.sum()

    summary = {
        "n_regions": int(n_regions),
        "n_policies": int(n_policies),
        "n_observed": int(n_obs),
        "observation_rate_pct": round(100 * n_obs / n_total, 2),
        "n_svd_regions": int(n_svd_regions),
        "singular_values": S.tolist(),
        "explained_variance_pct": (explained * 100).tolist(),
        "cumulative_variance_pct": (cumulative * 100).tolist(),
        "rank1_explains_pct": round(float(100 * explained[0]), 2),
        "rank1_r_squared_uncentered": round(float(rank1_r_squared), 4),
        "rank1_r_squared_centered": round(float(rank1_r_squared_centered), 4),
        "rank1_fit_rmse_observed": round(float(np.sqrt(obs_residuals.sum() / obs_mask.sum())), 6),
        "best_lambda": float(best_lambda),
        "effective_rank_completed": int(effective_rank),
        "completed_singular_values": S_completed.tolist(),
        "completed_explained_variance_pct": (s_explained_completed * 100).tolist(),
        "rank1_fit_mse": float(rank1_fit_mse),
        "lowrank_fit_mse": float(lowrank_fit_mse),
        "holdout_rank1_rmse": float(np.sqrt(agg["rank1_sq_error"])),
        "holdout_lowrank_rmse": float(np.sqrt(agg["lowrank_sq_error"])),
        "holdout_rank1_mae": float(agg["rank1_error"]),
        "holdout_lowrank_mae": float(agg["lowrank_error"]),
        "holdout_rmse_improvement_pct": round(float(100 * improvement_rmse), 2),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    gamma_df = pd.DataFrame(gamma_matrix, index=region_ids, columns=policy_names)
    gamma_df.to_csv(output_dir / "gamma_matrix_observed.csv")

    rank1_df = pd.DataFrame(rank1_completed, index=region_ids, columns=policy_names)
    rank1_df.to_csv(output_dir / "gamma_matrix_rank1.csv")

    lowrank_df = pd.DataFrame(X_lowrank, index=region_ids, columns=policy_names)
    lowrank_df.to_csv(output_dir / "gamma_matrix_lowrank.csv")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Rank-1 R^2 on observed entries (centered): {summary['rank1_r_squared_centered']:.4f}")
    print(f"  First singular value explains {summary['rank1_explains_pct']:.2f}% of SVD variance")
    print(f"  Holdout RMSE improvement from low-rank: {summary['holdout_rmse_improvement_pct']:.1f}%")

    r2 = summary["rank1_r_squared_centered"]
    if r2 >= 0.85:
        print("  >>> CONCLUSION: Rank-1 (separability) assumption is WELL SUPPORTED.")
        print("      The rank-1 model gamma_{R,i} = k_R * gamma_i explains")
        print(f"      {100*r2:.1f}% of variance in observed entries.")
    elif r2 >= 0.70:
        print("  >>> CONCLUSION: Rank-1 captures most variance; low-rank may help marginally.")
    else:
        print("  >>> CONCLUSION: Rank-1 leaves substantial residual variance.")
        print("      Low-rank generalization recommended.")

    print(f"\n  All outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
