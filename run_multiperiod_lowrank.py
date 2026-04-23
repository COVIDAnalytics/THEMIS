"""
Multi-period low-rank validation of the separability assumption.

For each of three pandemic periods, compute:
  - Holdout RMSE at ranks 1, 2, 3
  - Scatter plots of rank-1 vs rank-2 and rank-1 vs rank-3 imputed gammas

Outputs a single results table suitable for the Authors_response.tex.
"""
import json
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
    POLICY_INDEX,
    build_gamma_matrix,
    rank1_imputation,
)
from compare_rank1_vs_rank3 import rank_r_imputation

OUTPUT_DIR = Path("simulation_results/multiperiod_lowrank")

PERIODS = [
    ("2020-03-15", "2020-06-15", "2020.03.15--2020.06.15"),
    ("2020-06-15", "2020-09-15", "2020.06.15--2020.09.15"),
    ("2020-09-15", "2020-12-15", "2020.09.15--2020.12.15"),
]

RANKS_TO_TEST = [1, 2, 3]


def holdout_multirank(
    gamma_matrix: np.ndarray,
    obs_mask: np.ndarray,
    ranks: List[int],
    n_splits: int = 20,
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Holdout RMSE and MAE for each rank in `ranks`.
    Returns {rank: (rmse, rmse_std, mae)}.
    """
    rng = np.random.default_rng(seed)
    obs_indices = np.argwhere(obs_mask)
    n_obs = len(obs_indices)
    n_holdout = max(1, int(n_obs * holdout_frac))

    per_split = {r: [] for r in ranks}

    for split in range(n_splits):
        perm = rng.permutation(n_obs)
        ho_idx = obs_indices[perm[:n_holdout]]
        train_mask = obs_mask.copy()
        for ri, ci in ho_idx:
            train_mask[ri, ci] = False

        completions = {}
        r1_comp, _ = rank1_imputation(gamma_matrix, train_mask)
        completions[1] = r1_comp
        for rank in sorted(ranks):
            if rank == 1:
                continue
            ps = 0.1 if rank >= 3 else 0.0
            mi = 2000 if rank >= 3 else 500
            comp = rank_r_imputation(gamma_matrix, train_mask, rank=rank,
                                      warm_start=r1_comp,
                                      perturb_scale=ps,
                                      max_iter=mi,
                                      seed=42 + split * 100 + rank)
            completions[rank] = comp

        for rank in ranks:
            sq_errs = [(completions[rank][ri, ci] - gamma_matrix[ri, ci]) ** 2
                       for ri, ci in ho_idx]
            per_split[rank].append(np.sqrt(np.mean(sq_errs)))

    results = {}
    for rank in ranks:
        splits = np.array(per_split[rank])
        results[rank] = (splits.mean(), splits.std(), 0.0)
    return results


def plot_scatter_combined(
    period_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    output_dir: Path,
) -> None:
    """
    2-row x 3-col scatter figure comparing imputed entries only:
    rank-1 vs rank-2 (row 1) and rank-1 vs rank-3 (row 2).

    Parameters
    ----------
    period_data : list of (rank1_completed, rank2_completed, rank3_completed,
                           obs_mask, short_label)
    """
    n_periods = len(period_data)
    fig, axes = plt.subplots(2, n_periods, figsize=(5.0 * n_periods, 9.0))
    if n_periods == 1:
        axes = axes.reshape(2, 1)

    period_short = ["Period 1", "Period 2", "Period 3"]

    comparisons = [
        (1, 2, "#1b9e77"),
        (1, 3, "#d95f02"),
    ]

    for row, (ra, rb, color) in enumerate(comparisons):
        for col, (r1, r2, r3, mask, label) in enumerate(period_data):
            a_data = r1
            b_data = r2 if rb == 2 else r3

            imp_a, imp_b = a_data[~mask], b_data[~mask]

            ax = axes[row, col]
            ax.scatter(imp_a, imp_b, alpha=0.25, s=10, c=color,
                       edgecolors="none")
            lo = min(imp_a.min(), imp_b.min()) - 0.05
            hi = max(imp_a.max(), imp_b.max()) + 0.05
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            corr = np.corrcoef(imp_a, imp_b)[0, 1]
            ax.text(
                0.95, 0.08, f"$r = {corr:.3f}$",
                transform=ax.transAxes, fontsize=13, ha="right",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray",
                          pad=2),
            )
            ax.set_title(
                f"Imputed — {period_short[col]}",
                fontsize=14, fontweight="bold",
            )
            ax.set_xlabel(f"Rank-{ra} $\\gamma_{{R,i}}$", fontsize=13)
            ax.tick_params(labelsize=11)
            if col == 0:
                ax.set_ylabel(f"Rank-{rb} $\\gamma_{{R,i}}$", fontsize=13)

    fig.text(0.01, 0.75, "Rank-1 vs Rank-2", fontsize=16, fontweight="bold",
             rotation=90, va="center", ha="center")
    fig.text(0.01, 0.28, "Rank-1 vs Rank-3", fontsize=16, fontweight="bold",
             rotation=90, va="center", ha="center")

    fig.tight_layout(h_pad=3.0, w_pad=2.5, rect=[0.03, 0, 1, 1])
    fig.savefig(output_dir / "scatter_rank1_vs_rank2_rank3_combined.pdf",
                bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "scatter_rank1_vs_rank2_rank3_combined.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MULTI-PERIOD LOW-RANK VALIDATION")
    print("=" * 70)

    all_rows = []
    scatter_data = []  # collect (r1, r2, r3, mask, label) for combined plot

    for start, end, label in PERIODS:
        print(f"\n{'=' * 60}")
        print(f"  Period: {label}")
        print(f"{'=' * 60}")

        print(f"  Building gamma matrix ({start} to {end}) ...")
        gamma_matrix, obs_mask, region_ids, _ = build_gamma_matrix(
            start_date=start, end_date=end, min_policy_days=10,
        )
        n_regions, n_policies = gamma_matrix.shape
        n_obs = int(obs_mask.sum())
        obs_rate = 100 * n_obs / (n_regions * n_policies)
        obs_per_region = obs_mask.sum(axis=1)
        print(f"  {n_regions} regions x {n_policies} policies, "
              f"{n_obs} observed ({obs_rate:.1f}%)")
        print(f"  Obs/region: mean={obs_per_region.mean():.1f}, "
              f"median={np.median(obs_per_region):.0f}, "
              f"<=2: {(obs_per_region <= 2).sum()}/{n_regions}")

        if n_obs < 20:
            print(f"  WARNING: Too few observations ({n_obs}), skipping period.")
            for rank in RANKS_TO_TEST:
                all_rows.append({
                    "period": label,
                    "rank": rank,
                    "holdout_rmse": np.nan,
                    "holdout_mae": np.nan,
                    "n_regions": n_regions,
                    "n_obs": n_obs,
                    "obs_rate_pct": obs_rate,
                })
            continue

        # --- Holdout RMSE for each rank ---
        print(f"  Holdout RMSE (ranks {RANKS_TO_TEST}, 20 splits, 20% holdout) ...")
        ho_results = holdout_multirank(
            gamma_matrix, obs_mask, RANKS_TO_TEST,
            n_splits=20, holdout_frac=0.2, seed=42,
        )

        for rank in RANKS_TO_TEST:
            rmse, rmse_std, _ = ho_results[rank]
            all_rows.append({
                "period": label,
                "rank": rank,
                "holdout_rmse": round(rmse, 4),
                "holdout_rmse_std": round(rmse_std, 4),
                "n_regions": n_regions,
                "n_obs": n_obs,
                "obs_rate_pct": round(obs_rate, 1),
            })
            print(f"    Rank {rank}: RMSE = {rmse:.4f} ± {rmse_std:.4f}")

        # --- Collect scatter data (all three ranks) ---
        r1_completed, _ = rank1_imputation(gamma_matrix, obs_mask)
        r2_completed = rank_r_imputation(gamma_matrix, obs_mask, rank=2,
                                          warm_start=r1_completed)
        r3_completed = rank_r_imputation(gamma_matrix, obs_mask, rank=3,
                                          warm_start=r1_completed,
                                          perturb_scale=0.1,
                                          max_iter=2000)
        scatter_data.append((r1_completed, r2_completed, r3_completed,
                             obs_mask, label))

    # --- Combined scatter plot (4 rows x 3 cols: r1 vs r2 + r1 vs r3) ---
    if scatter_data:
        print(f"\n  Generating combined scatter plot ...")
        plot_scatter_combined(scatter_data, OUTPUT_DIR)

    # --- Combined results table ---
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTPUT_DIR / "multiperiod_results.csv", index=False)

    # Pivot for LaTeX-ready format
    pivot = results_df.pivot_table(
        index="period",
        columns="rank",
        values=["holdout_rmse"],
    )
    pivot.columns = [f"{metric}_rank{rank}" for metric, rank in pivot.columns]
    pivot = pivot.reset_index()
    pivot.to_csv(OUTPUT_DIR / "multiperiod_pivot.csv", index=False)

    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\n  Pivot table:")
    print(pivot.to_string(index=False))

    with open(OUTPUT_DIR / "multiperiod_summary.json", "w") as f:
        json.dump({
            "periods": [
                {
                    "label": row["period"],
                    "rank": int(row["rank"]),
                    "holdout_rmse": row["holdout_rmse"],
                    "holdout_rmse_std": row["holdout_rmse_std"],
                    "n_regions": int(row["n_regions"]),
                    "n_obs": int(row["n_obs"]),
                }
                for _, row in results_df.iterrows()
            ],
        }, f, indent=2)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
