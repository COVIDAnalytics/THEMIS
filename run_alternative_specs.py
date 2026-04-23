"""
Compare alternative specifications for gamma_{R,i} against the rank-1 model.

Models:
  - Rank-1 (THEMIS baseline):   gamma_{R,i} = k_R * gamma_i
  - Rank-1 + FE (Athey 2021):   gamma_{R,i} = u_R * v_i + alpha_R + beta_i
  - Additive two-way FE:        gamma_{R,i} = alpha_R + beta_i
  - Policy-only (homogeneous):  gamma_{R,i} = beta_i

Outputs:
  - Holdout RMSE table across three pandemic periods
  - Bar chart comparing holdout RMSE
"""
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_gamma_rank import (
    POLICY_NAMES,
    N_POLICIES,
    build_gamma_matrix,
    rank1_imputation,
)

OUTPUT_DIR = Path("simulation_results/alternative_specs")

PERIODS = [
    ("2020-03-15", "2020-06-15", "2020.03.15--2020.06.15"),
    ("2020-06-15", "2020-09-15", "2020.06.15--2020.09.15"),
    ("2020-09-15", "2020-12-15", "2020.09.15--2020.12.15"),
]


def additive_fe_imputation(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Additive two-way fixed effects: gamma_{R,i} = alpha_R + beta_i.

    Alternating estimation on observed entries only, with non-negative
    projection on the completed matrix.
    """
    n_R, n_I = gamma_matrix.shape

    beta = np.zeros(n_I)
    for j in range(n_I):
        obs_j = obs_mask[:, j]
        if obs_j.any():
            beta[j] = gamma_matrix[obs_j, j].mean()

    alpha = np.zeros(n_R)

    for _ in range(max_iter):
        old_alpha, old_beta = alpha.copy(), beta.copy()

        for r in range(n_R):
            obs_r = obs_mask[r, :]
            if obs_r.any():
                alpha[r] = (gamma_matrix[r, obs_r] - beta[obs_r]).mean()
            else:
                alpha[r] = 0.0

        for j in range(n_I):
            obs_j = obs_mask[:, j]
            if obs_j.any():
                beta[j] = (gamma_matrix[obs_j, j] - alpha[obs_j]).mean()
            else:
                beta[j] = 0.0

        change = np.sqrt(np.sum((alpha - old_alpha) ** 2)
                         + np.sum((beta - old_beta) ** 2))
        if change < tol:
            break

    completed = alpha[:, None] + beta[None, :]
    np.clip(completed, 0, None, out=completed)
    completed[obs_mask] = gamma_matrix[obs_mask]
    return completed


def policy_only_imputation(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
) -> np.ndarray:
    """
    Policy-only (homogeneous) model: gamma_{R,i} = beta_i.

    beta_i is the mean of observed gamma values for policy i across regions.
    Assumes no region heterogeneity.
    """
    n_R, n_I = gamma_matrix.shape
    beta = np.zeros(n_I)
    for j in range(n_I):
        obs_j = obs_mask[:, j]
        if obs_j.any():
            beta[j] = gamma_matrix[obs_j, j].mean()

    completed = np.tile(beta, (n_R, 1))
    np.clip(completed, 0, None, out=completed)
    completed[obs_mask] = gamma_matrix[obs_mask]
    return completed


def proper_rank1_imputation(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
) -> np.ndarray:
    """Delegates to rank1_imputation (ALS with default-gamma initialization)."""
    completed, _ = rank1_imputation(gamma_matrix, obs_mask)
    return completed


def rank1_plus_fe_imputation(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Rank-1 + additive fixed effects (Athey et al. 2021 style):
        gamma_{R,i} = u_R * v_i  +  alpha_R  +  beta_i

    Alternates between estimating the additive component (alpha, beta) and
    the rank-1 multiplicative component via truncated SVD on the residuals.
    """
    n_R, n_I = gamma_matrix.shape

    alpha = np.zeros(n_R)
    beta = np.zeros(n_I)
    for j in range(n_I):
        oj = obs_mask[:, j]
        if oj.any():
            beta[j] = gamma_matrix[oj, j].mean()

    L = np.zeros((n_R, n_I))

    for outer in range(max_iter):
        old_alpha, old_beta = alpha.copy(), beta.copy()
        old_L = L.copy()

        for r in range(n_R):
            obs_r = obs_mask[r, :]
            if obs_r.any():
                alpha[r] = (gamma_matrix[r, obs_r] - L[r, obs_r] - beta[obs_r]).mean()
            else:
                alpha[r] = 0.0

        for j in range(n_I):
            oj = obs_mask[:, j]
            if oj.any():
                beta[j] = (gamma_matrix[oj, j] - L[oj, j] - alpha[oj]).mean()
            else:
                beta[j] = 0.0

        resid = np.zeros_like(gamma_matrix)
        for r in range(n_R):
            for j in range(n_I):
                if obs_mask[r, j]:
                    resid[r, j] = gamma_matrix[r, j] - alpha[r] - beta[j]
                else:
                    resid[r, j] = L[r, j]

        U, S, Vt = np.linalg.svd(resid, full_matrices=False)
        L = S[0] * np.outer(U[:, 0], Vt[0, :])

        change = (np.sqrt(np.sum((alpha - old_alpha) ** 2)
                          + np.sum((beta - old_beta) ** 2))
                  + np.linalg.norm(L - old_L, 'fro') / max(np.linalg.norm(L, 'fro'), 1e-12))
        if change < tol:
            break

    completed = L + alpha[:, None] + beta[None, :]
    np.clip(completed, 0, None, out=completed)
    completed[obs_mask] = gamma_matrix[obs_mask]
    return completed


def region_only_imputation(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
) -> np.ndarray:
    """
    Region-only model: gamma_{R,i} = alpha_R.

    alpha_R is the mean of observed gamma values for region R across policies.
    Assumes no policy heterogeneity.
    """
    n_R, n_I = gamma_matrix.shape
    alpha = np.zeros(n_R)
    for r in range(n_R):
        obs_r = obs_mask[r, :]
        if obs_r.any():
            alpha[r] = gamma_matrix[r, obs_r].mean()

    completed = np.tile(alpha[:, None], (1, n_I))
    np.clip(completed, 0, None, out=completed)
    completed[obs_mask] = gamma_matrix[obs_mask]
    return completed


MODELS = {
    "Rank-1":       lambda gm, om: proper_rank1_imputation(gm, om),
    "Policy-only":  lambda gm, om: policy_only_imputation(gm, om),
    "Region-only":  lambda gm, om: region_only_imputation(gm, om),
}


def holdout_comparison(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    n_splits: int = 20,
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Holdout RMSE and MAE for each model. Returns {name: (rmse, rmse_std)}."""
    rng = np.random.default_rng(seed)
    obs_indices = np.argwhere(obs_mask)
    n_obs = len(obs_indices)
    n_holdout = max(1, int(n_obs * holdout_frac))

    per_split = {name: [] for name in MODELS}

    for split in range(n_splits):
        perm = rng.permutation(n_obs)
        ho_idx = obs_indices[perm[:n_holdout]]
        train_mask = obs_mask.copy()
        for ri, ci in ho_idx:
            train_mask[ri, ci] = False

        completions = {}
        for name, impute_fn in MODELS.items():
            completions[name] = impute_fn(gamma_matrix, train_mask)

        for name in MODELS:
            sq_errs = [(completions[name][ri, ci] - gamma_matrix[ri, ci]) ** 2
                       for ri, ci in ho_idx]
            per_split[name].append(np.sqrt(np.mean(sq_errs)))

    results = {}
    for name in MODELS:
        splits = np.array(per_split[name])
        results[name] = (splits.mean(), splits.std())
    return results


def plot_rmse_comparison(
    all_results: List[Tuple[str, Dict[str, Tuple[float, float]]]],
    output_dir: Path,
) -> None:
    """Grouped bar chart: holdout RMSE across models and periods."""
    model_names = list(MODELS.keys())
    n_models = len(model_names)
    n_periods = len(all_results)
    period_labels = [r[0] for r in all_results]

    short_labels = ["Period 1", "Period 2", "Period 3"]

    colors = ["#2c7fb8", "#d95f02", "#7570b3"]

    x = np.arange(n_periods)
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for m_idx, name in enumerate(model_names):
        vals = [all_results[p][1][name][0] for p in range(n_periods)]
        stds = [all_results[p][1][name][1] for p in range(n_periods)]
        offset = (m_idx - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=stds, capsize=3,
                      label=name, color=colors[m_idx],
                      edgecolor="white", linewidth=0.5,
                      error_kw=dict(lw=1, capthick=1))
        for bar, v, s in zip(bars, vals, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=12)
    ax.set_ylabel("Holdout RMSE", fontsize=13)
    ax.set_title("Alternative Specifications: Holdout RMSE Comparison", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(output_dir / "alt_specs_rmse_comparison.pdf",
                bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "alt_specs_rmse_comparison.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Bar chart saved to {output_dir / 'alt_specs_rmse_comparison.png'}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  ALTERNATIVE SPECIFICATIONS COMPARISON")
    print("=" * 70)

    all_results = []

    for start, end, label in PERIODS:
        print(f"\n{'=' * 60}")
        print(f"  Period: {label}")
        print(f"{'=' * 60}")

        gamma_matrix, obs_mask, region_ids, _ = build_gamma_matrix(
            start_date=start, end_date=end, min_policy_days=10,
        )
        n_regions, n_policies = gamma_matrix.shape
        n_obs = int(obs_mask.sum())
        obs_rate = 100 * n_obs / (n_regions * n_policies)
        print(f"  {n_regions} regions x {n_policies} policies, "
              f"{n_obs} observed ({obs_rate:.1f}%)")

        results = holdout_comparison(gamma_matrix, obs_mask)
        all_results.append((label, results))

        for name, (rmse, rmse_std) in results.items():
            print(f"    {name:25s}: RMSE = {rmse:.4f} ± {rmse_std:.4f}")

    print(f"\n{'=' * 60}")
    print("  Generating bar chart ...")
    plot_rmse_comparison(all_results, OUTPUT_DIR)

    print("\n  RESULTS SUMMARY:")
    header = f"  {'Model':25s}"
    for label, _ in all_results:
        header += f"  {label}"
    print(header)
    for name in MODELS:
        row = f"  {name:25s}"
        for _, results in all_results:
            rmse = results[name][0]
            row += f"  {rmse:>24.4f}"
        print(row)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
