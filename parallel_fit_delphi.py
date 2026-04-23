"""
Parallel DELPHI fitting for all regions using annealing.

Fits DELPHI on train_start..train_end for every discoverable region,
builds the gamma matrix from the fresh fits, runs rank-1 ALS, and
saves everything so the benchmark can load the cached results.

Usage:
    python parallel_fit_delphi.py --workers 20
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CASE_DATA_DIR = Path("pandemic_functions/pandemic_data")
_SKIP_REGIONS = {
    "Antarctica", "Diamond Princess", "MS Zaandam",
    "Summer Olympics 2020", "Winter Olympics 2022",
    "Western Sahara", "Holy See",
}

# ── region discovery (copied from benchmark for isolation) ────────────────

def _parse_case_file(path: Path) -> Optional[Tuple[str, str]]:
    try:
        row = pd.read_csv(path, nrows=1, keep_default_na=False)
        return str(row["country"].iloc[0]).strip(), str(row["province"].iloc[0]).strip() or "None"
    except Exception:
        return None


def _discover_all_regions() -> List[Tuple[str, str, Path]]:
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


def _region_id(country: str, province: str) -> str:
    return country if province == "None" else f"{country} - {province}"


# ── worker function (runs in a subprocess) ────────────────────────────────

def _fit_one_region(args_tuple):
    """Fit DELPHI for a single region. Returns dict with params or error."""
    country, province, case_path_str, train_start_str, train_end_str, eval_end_str, opt_method = args_tuple
    case_path = Path(case_path_str)
    train_start = pd.to_datetime(train_start_str).to_pydatetime()
    train_end = pd.to_datetime(train_end_str).to_pydatetime()
    eval_end = pd.to_datetime(eval_end_str).to_pydatetime()

    rid = _region_id(country, province)
    t0 = time.time()

    try:
        from pandemic_functions.pandemic_params import (
            global_populations,
            region_symbol_country_dict,
            region_symbol_continent_dict,
        )
        from pandemic_functions.delphi_functions.DELPHI_model_fitting import solve_and_predict_area

        # Register region
        region_symbol_country_dict[rid] = (country, province)
        pop_row = global_populations[
            (global_populations.Country == country) & (global_populations.Province == province)
        ]
        if pop_row.empty:
            return {"region": rid, "status": "no_population"}
        region_symbol_continent_dict[rid] = str(pop_row.iloc[0].get("Continent", "Unknown"))

        # Load case data
        cases = pd.read_csv(case_path, keep_default_na=False)
        cases["date"] = pd.to_datetime(cases["date"], errors="coerce").dt.normalize()
        cases = cases.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        for col in ["case_cnt", "death_cnt"]:
            cases[col] = pd.to_numeric(cases[col], errors="coerce").fillna(0)

        window = cases[(cases["date"] >= train_start) & (cases["date"] <= eval_end)].copy()
        if window.empty or window["case_cnt"].max() < 100:
            return {"region": rid, "status": "insufficient_cases"}

        fit_table = window.copy()
        fit_table["day_since100"] = (fit_table["date"] - train_start).dt.days.astype(int)
        for col in ["total_hospitalization", "people_vaccinated", "people_fully_vaccinated"]:
            if col not in fit_table.columns:
                fit_table[col] = 0

        yesterday = (train_end - timedelta(days=1)).strftime("%Y%m%d")
        df_params, _, df_pred, _ = solve_and_predict_area(
            region=rid,
            yesterday=yesterday,
            past_parameters=None,
            totalcases=fit_table.copy(),
            end_date=eval_end.strftime("%Y-%m-%d"),
            optimization_method=opt_method,
        )

        params_row = df_params.iloc[0]
        elapsed = time.time() - t0
        return {
            "region": rid,
            "country": country,
            "province": province,
            "status": "ok",
            "elapsed_s": round(elapsed, 1),
            "Infection Rate": float(params_row["Infection Rate"]),
            "Median Day of Action": float(params_row["Median Day of Action"]),
            "Rate of Action": float(params_row["Rate of Action"]),
            "Rate of Death": float(params_row["Rate of Death"]),
            "Mortality Rate": float(params_row["Mortality Rate"]),
            "Rate of Mortality Rate Decay": float(params_row["Rate of Mortality Rate Decay"]),
            "Internal Parameter 1": float(params_row["Internal Parameter 1"]),
            "Internal Parameter 2": float(params_row["Internal Parameter 2"]),
            "Jump Magnitude": float(params_row["Jump Magnitude"]),
            "Jump Time": float(params_row["Jump Time"]),
            "Jump Decay": float(params_row["Jump Decay"]),
            "MAPE": float(params_row.get("MAPE", -1)),
            "Data Start Date": train_start.strftime("%Y-%m-%d"),
        }
    except Exception as exc:
        elapsed = time.time() - t0
        return {"region": rid, "status": f"error:{type(exc).__name__}:{exc}", "elapsed_s": round(elapsed, 1)}


# ── gamma matrix from fresh fits ─────────────────────────────────────────

def _build_gamma_matrix_from_fresh_params(
    params_df: pd.DataFrame,
    train_start_str: str,
    train_end_str: str,
    min_policy_days: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build gamma matrix from freshly fitted params (same logic as build_gamma_matrix but with fresh fits)."""
    from pandemic_functions.pandemic_params import (
        default_dict_normalized_policy_gamma,
        future_policies,
        region_symbol_country_dict,
        region_symbol_continent_dict,
    )
    from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
        read_oxford_country_policy_data,
        read_policy_data_us_only,
    )
    from pandemic_functions.delphi_functions.DELPHI_utils import gamma_t

    policy_names = sorted(default_dict_normalized_policy_gamma.keys())
    n_policies = len(policy_names)
    policy_index = {p: i for i, p in enumerate(policy_names)}

    region_ids: List[str] = []
    gamma_rows: List[np.ndarray] = []
    obs_rows: List[np.ndarray] = []

    for _, prow in params_df.iterrows():
        country = str(prow["country"])
        province = str(prow["province"])
        rid = str(prow["region"])
        data_start_date = prow["Data Start Date"]

        # Register if needed
        region_symbol_country_dict[rid] = (country, province)
        pop_row_check = True  # already checked during fitting

        params_list = [
            data_start_date,
            float(prow["Median Day of Action"]),
            float(prow["Rate of Action"]),
            float(prow["Jump Magnitude"]),
            float(prow["Jump Time"]),
            float(prow["Jump Decay"]),
        ]

        # Read policy data
        try:
            if country == "US" and province != "None":
                pol = read_policy_data_us_only(
                    state=province,
                    start_date=train_start_str,
                    end_date=train_end_str,
                )
            else:
                pol = read_oxford_country_policy_data(
                    country=country,
                    start_date=train_start_str,
                    end_date=train_end_str,
                )
        except Exception:
            continue
        if pol is None or pol.empty:
            continue

        pol["date"] = pd.to_datetime(pol["date"], errors="coerce").dt.normalize()
        pol = pol.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        for p in future_policies:
            if p not in pol.columns:
                pol[p] = 0
            pol[p] = pol[p].fillna(0).astype(int)

        gamma_sum = {p: 0.0 for p in policy_names}
        gamma_count = {p: 0 for p in policy_names}
        for _, row in pol.iterrows():
            active = None
            for p in future_policies:
                if int(row.get(p, 0)) == 1:
                    active = p
                    break
            if active is None or active not in policy_index:
                continue
            g_val = float(gamma_t(row["date"], params_list))
            gamma_sum[active] += g_val
            gamma_count[active] += 1

        row_gamma = np.full(n_policies, np.nan)
        row_obs = np.zeros(n_policies, dtype=bool)
        has_any = False
        for p in policy_names:
            idx = policy_index[p]
            if gamma_count[p] >= min_policy_days:
                row_gamma[idx] = gamma_sum[p] / gamma_count[p]
                row_obs[idx] = True
                has_any = True

        if not has_any:
            continue

        gamma_rid = f"{country}__{province}".replace(" ", "_")
        region_ids.append(gamma_rid)
        gamma_rows.append(row_gamma)
        obs_rows.append(row_obs)

    return np.array(gamma_rows), np.array(obs_rows), region_ids, policy_names


def main():
    parser = argparse.ArgumentParser(description="Parallel DELPHI fitting for all regions")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--train-start", type=str, default="2020-03-15")
    parser.add_argument("--train-end", type=str, default="2020-06-15")
    parser.add_argument("--eval-end", type=str, default="2020-09-15")
    parser.add_argument("--optimization-method", type=str, default="annealing", choices=["annealing", "tnc"])
    parser.add_argument("--min-policy-days", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="simulation_results/fresh_fits")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_regions = _discover_all_regions()
    print(f"[INFO] Discovered {len(all_regions)} regions")

    # Build work items
    work = [
        (country, province, str(case_path), args.train_start, args.train_end, args.eval_end, args.optimization_method)
        for country, province, case_path in all_regions
    ]

    print(f"[FIT] Fitting {len(work)} regions with {args.optimization_method} using {args.workers} workers ...")
    t_start = time.time()

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_fit_one_region, w): w[0] + "/" + w[1] for w in work}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            status = result["status"]
            elapsed = result.get("elapsed_s", "?")
            if status == "ok":
                alpha = result["Infection Rate"]
                print(f"  [{done:>3d}/{len(work)}] {result['region']:40s} OK ({elapsed}s, alpha={alpha:.3f})")
            else:
                print(f"  [{done:>3d}/{len(work)}] {result['region']:40s} {status} ({elapsed}s)")

    total_time = time.time() - t_start
    ok_results = [r for r in results if r["status"] == "ok"]
    print(f"\n[FIT] Done in {total_time:.0f}s. {len(ok_results)}/{len(work)} regions fitted successfully.")

    # Save fitted params
    params_df = pd.DataFrame(ok_results)
    params_path = out_dir / "fresh_params.csv"
    params_df.to_csv(params_path, index=False)
    print(f"[SAVE] Params: {params_path}")

    # Build gamma matrix from fresh fits
    print(f"\n[GAMMA] Building gamma matrix from {len(ok_results)} fresh fits (min_policy_days={args.min_policy_days}) ...")
    gamma_matrix, obs_mask, gamma_region_ids, policy_names = _build_gamma_matrix_from_fresh_params(
        params_df=params_df,
        train_start_str=args.train_start,
        train_end_str=args.train_end,
        min_policy_days=args.min_policy_days,
    )
    n_regions_gm = len(gamma_region_ids)
    n_policies = len(policy_names)
    print(f"[GAMMA] {n_regions_gm} regions x {n_policies} policies, "
          f"{obs_mask.sum()} observed ({100*obs_mask.sum()/(n_regions_gm*n_policies):.1f}%)")

    # Rank-1 ALS
    from analyze_gamma_rank import rank1_imputation
    print("[ALS] Running rank-1 imputation ...")
    completed_matrix, k_vector = rank1_imputation(gamma_matrix, obs_mask)
    print(f"[ALS] Done. k_R range: [{k_vector.min():.4f}, {k_vector.max():.4f}]")

    # Save gamma data
    np.savez(
        out_dir / "gamma_data.npz",
        gamma_matrix=gamma_matrix,
        obs_mask=obs_mask,
        completed_matrix=completed_matrix,
        k_vector=k_vector,
    )
    with open(out_dir / "gamma_region_ids.json", "w") as f:
        json.dump(gamma_region_ids, f)
    with open(out_dir / "gamma_policy_names.json", "w") as f:
        json.dump(policy_names, f)

    # Save config
    config = {
        "train_start": args.train_start,
        "train_end": args.train_end,
        "eval_end": args.eval_end,
        "optimization_method": args.optimization_method,
        "min_policy_days": args.min_policy_days,
        "workers": args.workers,
        "total_regions_discovered": len(all_regions),
        "regions_fitted": len(ok_results),
        "regions_in_gamma_matrix": n_regions_gm,
        "total_fit_time_s": round(total_time, 1),
    }
    with open(out_dir / "fit_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[DONE] All outputs saved to {out_dir}/")
    print(f"  fresh_params.csv        ({len(ok_results)} regions)")
    print(f"  gamma_data.npz          ({n_regions_gm} regions x {n_policies} policies)")
    print(f"  gamma_region_ids.json")
    print(f"  gamma_policy_names.json")
    print(f"  fit_config.json")


if __name__ == "__main__":
    main()
