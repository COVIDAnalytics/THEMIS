"""
Sensitivity analysis: adherence decay / enforcement fatigue.

For each of 4 paper regions, enumerate all 7^3 = 343 policy sequences and
run DELPHI with constant gamma vs linearly-decaying gamma within each
monthly NPI block.  Report the Spearman rank correlation of the predicted-
cases ranking between constant and decaying specifications.
"""
import itertools
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import stats
from scipy.integrate import solve_ivp

from analyze_gamma_rank import (
    PARAM_COLS,
    POLICY_INDEX,
    POLICY_NAMES,
    N_POLICIES,
    build_gamma_matrix,
    rank1_imputation,
)
from pandemic_functions.pandemic_params import (
    region_symbol_continent_dict,
    region_symbol_country_dict,
    p_v, p_d, p_h,
    global_populations,
    validcases_threshold_policy,
    bounds_q,
)
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import past_parameters
from pandemic_functions.delphi_functions.DELPHI_model import model_covid
from pandemic_functions.delphi_functions.DELPHI_utils import (
    get_initial_conditions,
    create_datasets_with_confidence_intervals,
)
from policy_functions.policy import Policy
from run_themis_region_holdout import (
    _register_region,
    prepare_global_artifacts,
)
from compare_rank1_vs_rank3 import matrix_row_to_gamma_dict

warnings.filterwarnings("ignore")

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

NPI_LIST = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Mass_Gatherings_Authorized_But_Others_Restricted",
    "Restrict_Mass_Gatherings_and_Schools",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]

OUTPUT_DIR = Path("simulation_results/adherence_decay_sensitivity")
DECAY_RATES = [0.05, 0.10, 0.15, 0.20]


def run_delphi_with_decay(
    region: str,
    policy_vector: List[str],
    start_date_str: str,
    totalcases: pd.DataFrame,
    gamma_dict: Dict[str, float],
    decay_rate: float,
) -> Optional[Tuple[float, float]]:
    """Run DELPHI for a 3-month policy with optional per-day adherence decay.
    Returns (predicted_cases, predicted_deaths) or None."""
    country, province = region_symbol_country_dict[region]
    continent = region_symbol_continent_dict[region]
    param_rows = past_parameters[
        (past_parameters.Country == country) & (past_parameters.Province == province)
    ]
    if param_rows.empty:
        return None

    line = param_rows.iloc[-1].values.tolist()
    params = line[5:]
    date_day_since100 = pd.to_datetime(line[3])

    policy_start = pd.to_datetime(start_date_str)
    policy_end = policy_start + relativedelta(months=len(policy_vector))

    vc = totalcases[
        (totalcases.date >= str(date_day_since100.date()))
        & (totalcases.date <= str(policy_end.date()))
    ][["day_since100", "case_cnt", "death_cnt", "total_hospitalization",
       "people_vaccinated", "people_fully_vaccinated"]].reset_index(drop=True)

    if len(vc) <= validcases_threshold_policy:
        return None

    N = global_populations[
        (global_populations.Country == country) & (global_populations.Province == province)
    ].pop2016.iloc[-1]
    PopI = vc.loc[0, "case_cnt"]
    PopD = vc.loc[0, "death_cnt"]
    PopR = PopD * 5 if PopI - PopD > PopD * 5 else 0

    maxT = (policy_end - date_day_since100).days + 1
    if policy_start < date_day_since100:
        return None
    pStartT = (policy_start - date_day_since100).days + 1
    GFIX = (N, PopR, PopD, PopI, p_v, p_d, p_h)
    t_pred = list(range(maxT))

    gammas = {}
    for i, npi in enumerate(policy_vector):
        bs = policy_start + relativedelta(months=i)
        be = policy_start + relativedelta(months=i + 1)
        base_g = gamma_dict[npi]
        if bs >= date_day_since100:
            t1 = (bs - date_day_since100).days + 1
            t2 = (be - date_day_since100).days + 1
            if decay_rate == 0.0:
                gammas[(t1, t2)] = base_g
            else:
                for d in range(t2 - t1):
                    td = t1 + d
                    gammas[(td, td + 1)] = min(base_g * (1.0 + decay_rate * d / 30.0), 2.0)

    x0 = get_initial_conditions(params_fitted=params, global_params_fixed=GFIX)

    def ode(t, x, a, dy, rs, rdth, pdth, rdd, k1, k2, j, tj, sn):
        return model_covid(t, x, a, dy, rs, rdth, pdth, rdd, k1, k2, j, tj, sn,
                           N, gammas, pStartT, maxT)

    sol = solve_ivp(ode, y0=x0, t_span=[t_pred[0], t_pred[-1]],
                    t_eval=t_pred, args=tuple(params)).y

    cases_fit = vc["case_cnt"].tolist()
    deaths_fit = vc["death_cnt"].tolist()
    yesterday = str((policy_start - timedelta(days=1)).date())

    df_pred, _ = create_datasets_with_confidence_intervals(
        continent, country, province, date_day_since100, yesterday,
        sol, cases_fit, deaths_fit, q=bounds_q)

    det = df_pred["Total Detected"].tolist()
    dth = df_pred["Total Detected Deaths"].tolist()
    return det[-1] - det[0], dth[-1] - dth[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  ADHERENCE DECAY: POLICY-RANKING SENSITIVITY")
    print("=" * 70)

    print("\n[1] Building gamma matrix + rank-1 imputation ...")
    gm, om, rids, _ = build_gamma_matrix(
        start_date="2020-03-15", end_date="2020-06-15", min_policy_days=10)
    r1, _ = rank1_imputation(gm, om)
    rid2idx = {r: i for i, r in enumerate(rids)}

    print("[2] Preparing DELPHI artifacts ...")
    arts = prepare_global_artifacts(
        start_date=datetime(2020, 3, 15), months=3, min_true_days=30, max_regions=0)

    all_sequences = list(itertools.product(NPI_LIST, repeat=3))
    n_seq = len(all_sequences)
    print(f"\n[3] Enumerating all {n_seq} policy sequences x 4 regions "
          f"x {1 + len(DECAY_RATES)} decay rates ...")

    results = {}

    for rid in sorted(PAPER_REGIONS.keys()):
        ri = rid2idx.get(rid)
        if ri is None or rid not in arts:
            print(f"    {rid}: not available, skipping")
            continue

        art = arts[rid]
        _register_region(rid, art["country"], art["province"], art["continent"])
        gdict = matrix_row_to_gamma_dict(r1[ri])
        display = PAPER_REGION_DISPLAY.get(rid, rid)
        results[rid] = {}

        for lam in [0.0] + DECAY_RATES:
            cases_list = []
            deaths_list = []
            n_fail = 0
            for seq in all_sequences:
                pv = list(seq)
                out = run_delphi_with_decay(
                    rid, pv, art["start_date_used"],
                    art["totalcases"], gdict, lam)
                if out is None:
                    cases_list.append(np.nan)
                    deaths_list.append(np.nan)
                    n_fail += 1
                else:
                    cases_list.append(out[0])
                    deaths_list.append(out[1])

            results[rid][lam] = {
                "cases": np.array(cases_list),
                "deaths": np.array(deaths_list),
            }
            ok = n_seq - n_fail
            print(f"    {display:>16s} | lam={lam:.2f} | {ok}/{n_seq} succeeded"
                  f" | {n_fail} failed")

    print("\n" + "=" * 70)
    print("  SPEARMAN RANK CORRELATIONS (constant vs decaying gamma)")
    print("=" * 70)

    summary_rows = []
    for rid in sorted(results.keys()):
        display = PAPER_REGION_DISPLAY.get(rid, rid)
        base_cases = results[rid][0.0]["cases"]
        base_deaths = results[rid][0.0]["deaths"]

        for lam in DECAY_RATES:
            dec_cases = results[rid][lam]["cases"]
            dec_deaths = results[rid][lam]["deaths"]

            valid = np.isfinite(base_cases) & np.isfinite(dec_cases)
            rho_c, p_c = stats.spearmanr(base_cases[valid], dec_cases[valid])

            valid_d = np.isfinite(base_deaths) & np.isfinite(dec_deaths)
            rho_d, p_d_val = stats.spearmanr(base_deaths[valid_d], dec_deaths[valid_d])

            summary_rows.append({
                "region_id": rid,
                "region": display,
                "decay_rate": lam,
                "rho_cases": rho_c,
                "p_cases": p_c,
                "rho_deaths": rho_d,
                "p_deaths": p_d_val,
                "n_valid": int(valid.sum()),
            })
            print(f"  {display:>16s} | lam={lam:.2f} | "
                  f"rho(cases)={rho_c:.6f}  rho(deaths)={rho_d:.6f}  "
                  f"(n={int(valid.sum())})")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "ranking_correlations.csv", index=False)

    with open(OUTPUT_DIR / "ranking_summary.json", "w") as f:
        json.dump({
            "decay_rates_tested": DECAY_RATES,
            "regions": list(PAPER_REGION_DISPLAY.values()),
            "correlations": [
                {k: (round(v, 6) if isinstance(v, float) else v)
                 for k, v in row.items()}
                for row in summary_rows
            ],
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("  DONE. Outputs in:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
