"""
True Online Rolling-Horizon NPI Advisor (v2 with rank-1 ALS gammas).

Hindsight gammas and all non-online simulations use the rank-1 ALS
pipeline built into Pandemic_Factory (see pandemic_functions/pandemic.py).

Online gammas at decision date t are obtained by (a) loading the
DELPHI parameter snapshot fitted only on data up to t, (b) building
the partially-observed cross-region gamma matrix on the window
[start_date, t] using that snapshot, (c) running rank-1 ALS to
impute missing entries, and (d) reading off the focal region's row.

The actual cost is taken from the OBSERVED data via Pandemic with
policy_type="actual".
"""
import argparse
import itertools
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandemic_functions.pandemic import Pandemic_Factory, Pandemic
from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.pandemic_params import region_symbol_country_dict
from pandemic_functions.delphi_functions import DELPHI_model_policy_scenarios as dmps
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    read_policy_data_us_only,
    read_oxford_country_policy_data,
)
from policy_functions.policy import Policy
from cost_functions.economic_cost.economic_data.economic_params import (
    TOTAL_GDP,
)
import analyze_gamma_rank as agr
from analyze_gamma_rank import build_gamma_matrix, rank1_imputation

REGION_POPULATION = {
    "DE": 83_240_000,
    "BR": 212_600_000,
    "ES": 47_350_000,
    "US-NY": 19_540_000,
}

TREE_BUNDLE_PATH = Path("simulation_results/state_action/state_action_trees.pkl")
TREE_POLICY_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]

FUTURE_POLICIES = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Mass_Gatherings_Authorized_But_Others_Restricted",
    "Restrict_Mass_Gatherings_and_Schools",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]

POLICY_SHORT = {
    "No_Measure": "None",
    "Restrict_Mass_Gatherings": "MG",
    "Mass_Gatherings_Authorized_But_Others_Restricted": "T+W",
    "Restrict_Mass_Gatherings_and_Schools": "MG+Sch",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others": "MG+T+W",
    "Restrict_Mass_Gatherings_and_Schools_and_Others": "MG+Sch+T+W",
    "Lockdown": "Lock",
}

POLICY_SEVERITY = {p: i for i, p in enumerate(FUTURE_POLICIES)}
POLICY_NUMBER = {p: i + 1 for i, p in enumerate(FUTURE_POLICIES)}

REGIONS = ["DE", "BR", "ES", "US-NY"]
START_DATE = "2020-03-15"

ROLLING_MONTHS = [
    ("2020-03-15", "2020-04-15"),
    ("2020-04-15", "2020-05-15"),
    ("2020-05-15", "2020-06-15"),
]

SNAPSHOT_BY_DECISION = {
    "2020-03-15": "pandemic_functions/pandemic_data/Parameters_Global_20200413.csv",
    "2020-04-15": "pandemic_functions/pandemic_data/Parameters_Global_20200515.csv",
    "2020-05-15": "pandemic_functions/pandemic_data/Parameters_Global_20200615.csv",
}

# ---------------------------------------------------------------------------
# Snapshot loading / V2 schema padding
# ---------------------------------------------------------------------------
def _load_snapshot_as_v2(snapshot_path):
    """Pad a pre-V2 DELPHI snapshot to the V2 column schema."""
    df = pd.read_csv(snapshot_path, keep_default_na=False)
    v2_cols = [
        "Continent", "Country", "Province", "Data Start Date", "MAPE",
        "Infection Rate", "Median Day of Action", "Rate of Action",
        "Rate of Death", "Mortality Rate", "Rate of Mortality Rate Decay",
        "Internal Parameter 1", "Internal Parameter 2",
        "Jump Magnitude", "Jump Time", "Jump Decay",
    ]
    defaults = {
        "MAPE": 10.0, "Rate of Death": 0.1,
        "Rate of Mortality Rate Decay": 0.0,
        "Jump Magnitude": 0.0, "Jump Time": 200.0, "Jump Decay": 1.0,
    }
    for col in v2_cols:
        if col not in df.columns:
            df[col] = defaults.get(col, 0.0)
    return df[v2_cols]


# ---------------------------------------------------------------------------
# Online gamma estimation: rank-1 ALS on the window data observable so far
# ---------------------------------------------------------------------------
def _region_matrix_key(region):
    """Map a region code to the key used in the gamma matrix (same logic
    as Pandemic_Factory._initialize_rank1)."""
    country, province = region_symbol_country_dict[region]
    return f"{country}__{province}".replace(" ", "_")


def _online_rank1_gammas(region, start_date, decision_date, snapshot_df):
    """Rank-1 ALS gammas using only data in [start_date, decision_date].

    Monkey-patch dmps.past_parameters AND analyze_gamma_rank.past_parameters
    with the dated snapshot, build the cross-region matrix, run ALS, and
    return the focal region's row.

    This date-windowed estimation cannot use the factory (whose init
    covers the full hindsight window), so we call build_gamma_matrix /
    rank1_imputation directly.
    """
    orig_dmps = dmps.past_parameters
    orig_agr = agr.past_parameters
    try:
        dmps.past_parameters = snapshot_df
        agr.past_parameters = snapshot_df

        gammas = None
        try:
            gamma_matrix, obs_mask, region_ids, policy_names = build_gamma_matrix(
                start_date=start_date, end_date=decision_date,
            )
            ok = (
                gamma_matrix.size > 0
                and gamma_matrix.ndim == 2
                and gamma_matrix.shape[0] >= 5
                and obs_mask.sum() > 0
            )
        except Exception:
            ok = False

        key = _region_matrix_key(region)
        if ok and key in region_ids:
            completed, _ = rank1_imputation(gamma_matrix, obs_mask)
            idx = region_ids.index(key)
            gammas = {p: float(completed[idx, j])
                      for j, p in enumerate(policy_names)}
            print(f"      [ALS] {gamma_matrix.shape[0]} regions, "
                  f"{obs_mask.sum()} observed entries")
        else:
            print(f"      [fallback] not enough policy data for ALS; "
                  f"using factory default gammas for initial month")
            gammas = None
    finally:
        dmps.past_parameters = orig_dmps
        agr.past_parameters = orig_agr
    return gammas


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
def _cost_dict(cost, policy_vector=None):
    """Extract a standard cost summary dict from a PandemicCost object."""
    econ = float(cost.st_economic_costs)
    humanitarian = float(cost.d_costs + cost.h_costs + cost.mh_costs)
    out = {
        "economic_costs": econ,
        "humanitarian_costs": humanitarian,
        "total_costs": econ + humanitarian,
        "num_cases": float(cost.num_cases),
        "num_deaths": float(cost.num_deaths),
        "d_costs": float(cost.d_costs),
        "h_costs": float(cost.h_costs),
        "mh_costs": float(cost.mh_costs),
    }
    if policy_vector is not None:
        out["policy_vector"] = policy_vector
    return out


def _evaluate_sequence(factory, region, policy_vector, start_date):
    """Evaluate a hypothetical sequence using the factory's stored rank-1
    ALS gammas (the standard hindsight evaluation path)."""
    try:
        policy = Policy(policy_type="hypothetical", start_date=start_date,
                        policy_vector=policy_vector)
        pandemic = factory.compute_delphi(policy, region=region)
        return _cost_dict(PandemicCost(pandemic), policy_vector)
    except Exception as e:
        print(f"    SKIP {policy_vector}: {type(e).__name__}: {e}")
        return None


def _evaluate_with_gammas(factory, region, policy_vector, start_date, gammas):
    """Evaluate a hypothetical sequence with an explicit gamma dict.

    Used for the online advisor where the gamma estimates are
    date-restricted and differ from the factory's stored values.
    """
    try:
        policy = Policy(policy_type="hypothetical", start_date=start_date,
                        policy_vector=policy_vector)
        country, province = region_symbol_country_dict[region]
        country_sub = country.replace(" ", "_")
        province_sub = province.replace(" ", "_")
        totalcases = pd.read_csv(
            f"pandemic_functions/pandemic_data/"
            f"Cases_{country_sub}_{province_sub}.csv"
        )
        pandemic = Pandemic(policy, region, factory.delphi_prediction,
                            totalcases, gammas)
        return _cost_dict(PandemicCost(pandemic), policy_vector)
    except Exception as e:
        print(f"    SKIP {policy_vector}: {type(e).__name__}: {e}")
        return None


def _real_actual_cost(factory, region, start_date, n_months):
    """Real actual cost obtained via policy_type='actual'."""
    policy = Policy(policy_type="actual", start_date=start_date,
                    policy_length=n_months)
    pandemic = factory.compute_delphi(policy, region=region)
    return _cost_dict(PandemicCost(pandemic))


def _get_actual_policy_for_month(region, month_start, month_end):
    """Identify the predominantly-active MECE policy for a month."""
    country, province = region_symbol_country_dict[region]
    try:
        if country == "US":
            policy_data = read_policy_data_us_only(
                state=province, start_date=month_start, end_date=month_end)
        else:
            policy_data = read_oxford_country_policy_data(
                country=country, start_date=month_start, end_date=month_end)
    except Exception:
        return "Unknown"
    if policy_data is None or len(policy_data) == 0:
        return "Unknown"

    policy_cols = [c for c in policy_data.columns
                   if c not in ["CountryName", "CountryCode", "Date",
                                "ConfirmedCases", "ConfirmedDeaths",
                                "date", "state"]]
    if not policy_cols:
        return "Unknown"
    day_counts = {}
    for _, row in policy_data.iterrows():
        active = [c for c in policy_cols if row.get(c, 0) == 1]
        label = active[-1] if active else "No_Measure"
        day_counts[label] = day_counts.get(label, 0) + 1
    return max(day_counts, key=day_counts.get)


def _online_advise(factory, region, context_policies, start_date,
                   online_gammas):
    """One-step lookahead with status-quo continuation:
    for each candidate next-month NPI, simulate that NPI for the next
    month and assume it stays in effect for the remaining months.
    Pick the cheapest by total cost under the online gammas.
    """
    results = []
    for npi in FUTURE_POLICIES:
        decided = context_policies + [npi]
        remaining = 3 - len(decided)
        policy_vector = decided + [npi] * remaining
        r = _evaluate_with_gammas(factory, region, policy_vector,
                                  start_date, online_gammas)
        if r is None:
            continue
        results.append({
            "npi": npi,
            "npi_short": POLICY_SHORT[npi],
            "economic_costs": r["economic_costs"],
            "humanitarian_costs": r["humanitarian_costs"],
            "total_costs": r["total_costs"],
            "num_deaths": r["num_deaths"],
            "full_vector": policy_vector,
        })
    if not results:
        return None
    df = pd.DataFrame(results).sort_values("total_costs")
    rec = df.iloc[0]
    return {
        "recommended_npi": rec["npi"],
        "recommended_short": rec["npi_short"],
        "recommended_cost": float(rec["total_costs"]),
        "ranking": df.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Decision-tree policy: a feasible online policy that queries the pre-trained
# state-action tree at each decision date.
# ---------------------------------------------------------------------------
def _load_tree_bundle(path=TREE_BUNDLE_PATH):
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _state_at_prefix(factory, region, prefix, start_date):
    """State features at the start of the next month after `prefix` has
    been executed.  Uses the factory's stored rank-1 gammas.  Mirrors
    `_state_at_decision` in `prescriptive_state_action.py` so the
    resulting state vector is a valid input to the trained tree."""
    pop = REGION_POPULATION[region]
    monthly_gdp = TOTAL_GDP[region] / 12.0
    if len(prefix) == 0:
        return {
            "last_severity": 0,
            "mean_prefix_severity": 0,
            "pop_pct_cum_cases": 0.0,
            "pop_pct_cum_deaths": 0.0,
            "pop_pct_active_cases": 0.0,
            "pop_pct_active_hosp": 0.0,
            "cum_econ_cost_pct_gdp": 0.0,
            "cum_human_cost_pct_gdp": 0.0,
        }
    policy = Policy(policy_type="hypothetical", start_date=start_date,
                    policy_vector=prefix)
    pandemic = factory.compute_delphi(policy, region=region)
    cost = PandemicCost(pandemic)
    def _safe(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if not np.isfinite(v) else v

    econ = _safe(cost.st_economic_costs)
    human = _safe(cost.d_costs) + _safe(cost.h_costs) + _safe(cost.mh_costs)

    return {
        "last_severity": POLICY_SEVERITY[prefix[-1]],
        "mean_prefix_severity": float(np.mean(
            [POLICY_SEVERITY[p] for p in prefix])),
        "pop_pct_cum_cases": 100.0 * _safe(cost.num_cases) / pop,
        "pop_pct_cum_deaths": 100.0 * _safe(cost.num_deaths) / pop,
        "pop_pct_active_cases": 100.0 * _safe(getattr(cost, "active_cases_end", 0.0)) / pop,
        "pop_pct_active_hosp": 100.0 * _safe(getattr(cost, "active_hosp_end", 0.0)) / pop,
        "cum_econ_cost_pct_gdp": 100.0 * econ / monthly_gdp,
        "cum_human_cost_pct_gdp": 100.0 * human / monthly_gdp,
    }


def _build_tree_policy_sequence(region, w, factory, tree_bundle, start_date):
    """Construct the 3-month NPI sequence prescribed by the pooled tree
    when queried month-by-month with the state induced by the tree's own
    earlier choices.  Uses the factory's stored rank-1 gammas for state
    computation.  The tree itself is offline (one-off training), so
    the resulting sequence is a feasible online policy."""
    pooled = tree_bundle["trees"]["pooled"]
    feature_names = pooled["feature_names"]
    classes = pooled["classes"]
    sklearn_tree = pooled["tree"]

    prefix = []
    for _ in range(3):
        state = _state_at_prefix(factory, region, prefix, start_date)
        state["w_humanitarian"] = w
        x = pd.DataFrame([[state[c] for c in feature_names]],
                         columns=feature_names)
        pred = sklearn_tree.predict(x)[0]
        prefix.append(pred)
    return prefix


# ---------------------------------------------------------------------------
# Main rolling-horizon driver
# ---------------------------------------------------------------------------
def run_online_rolling_v2(regions, start_date, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factory = Pandemic_Factory()
    factory._initialize_rank1()
    print(f"  Rank-1 ALS gammas loaded for "
          f"{len(factory.d_region_policy_gammas)} regions")
    all_results = {}

    snapshots = {d: _load_snapshot_as_v2(p)
                 for d, p in SNAPSHOT_BY_DECISION.items()}

    tree_bundle = _load_tree_bundle()
    if tree_bundle is None:
        print(f"  [warn] tree bundle not found at {TREE_BUNDLE_PATH}; "
              f"tree-policy will be skipped")

    for region in regions:
        print(f"\n{'='*60}")
        print(f"  Online Rolling-Horizon Advisor v2: {region}")
        print(f"{'='*60}")

        monthly = []
        online_context = []

        for month_idx, (m_start, m_end) in enumerate(ROLLING_MONTHS):
            print(f"\n  Month {month_idx+1}: decision at {m_start}")
            decision_date = m_start

            print(f"    Estimating online rank-1 gammas with snapshot "
                  f"{Path(SNAPSHOT_BY_DECISION[decision_date]).name} "
                  f"on window [{start_date}, {decision_date}] ...")
            online_gammas = _online_rank1_gammas(
                region, start_date, decision_date,
                snapshots[decision_date])
            if online_gammas is None:
                print(f"    Online ALS failed; falling back to "
                      f"factory rank-1 gammas")
                online_gammas = factory.d_region_policy_gammas[region]
            print("    Online gammas: " + ", ".join(
                f"{POLICY_SHORT.get(p,p)}={g:.3f}"
                for p, g in online_gammas.items()))

            actual_npi = _get_actual_policy_for_month(region, m_start, m_end)
            print(f"    Actual policy implemented: "
                  f"{POLICY_SHORT.get(actual_npi, actual_npi)}")

            advice = _online_advise(factory, region, online_context,
                                    start_date, online_gammas)
            if advice is None:
                print("    Advisor failed for this month")
                online_context.append(
                    actual_npi if actual_npi in FUTURE_POLICIES
                    else FUTURE_POLICIES[0])
                monthly.append({"month": month_idx+1, "error": True})
                continue

            rec = advice["recommended_npi"]
            print(f"    Online advisor recommends: {POLICY_SHORT[rec]} "
                  f"(cost={advice['recommended_cost']:.2e})")
            print("    Full ranking:")
            for r in advice["ranking"]:
                marker = ""
                if r["npi"] == rec:
                    marker += " <-- ONLINE"
                if r["npi"] == actual_npi:
                    marker += " <-- ACTUAL"
                print(f"      {r['npi_short']:>12}: total={r['total_costs']:.2e}"
                      f"{marker}")

            monthly.append({
                "month": month_idx+1,
                "decision_date": decision_date,
                "actual_npi": actual_npi,
                "actual_npi_short": POLICY_SHORT.get(actual_npi, actual_npi),
                "online_npi": rec,
                "online_npi_short": POLICY_SHORT[rec],
                "online_cost": advice["recommended_cost"],
                "online_gammas": {p: float(g) for p, g in online_gammas.items()},
                "ranking": advice["ranking"],
            })
            online_context.append(rec)

        online_seq = [m.get("online_npi") for m in monthly if "online_npi" in m]
        actual_seq = [m.get("actual_npi") for m in monthly]

        print(f"\n  --- 3-month sequences for {region} ---")
        print(f"    Online advisor: "
              f"{' -> '.join(POLICY_SHORT.get(p,p) for p in online_seq)}")
        print(f"    Actual policy:  "
              f"{' -> '.join(POLICY_SHORT.get(p,p) for p in actual_seq)}")

        online_3mo = _evaluate_sequence(
            factory, region, online_seq, start_date)

        real_actual = _real_actual_cost(factory, region, start_date,
                                        n_months=3)

        n_seqs = len(FUTURE_POLICIES) ** 3
        print(f"    Computing hindsight costs for all {n_seqs} sequences ...")
        hindsight_results = []
        for pv in itertools.product(FUTURE_POLICIES, repeat=3):
            r = _evaluate_sequence(factory, region, list(pv), start_date)
            if r is not None:
                hindsight_results.append(r)
        hindsight_optimal = None
        if hindsight_results:
            hdf = pd.DataFrame(hindsight_results)
            hindsight_optimal = hdf.loc[hdf["total_costs"].idxmin()].to_dict()

        tree_policy_seq = None
        tree_policy_cost = None
        tree_policy_weight_used = None
        tree_policy_sweep = []
        if tree_bundle is not None:
            print(f"    Sweeping decision-tree policy over "
                  f"w in {TREE_POLICY_WEIGHTS} ...")
            for w in TREE_POLICY_WEIGHTS:
                try:
                    seq_w = _build_tree_policy_sequence(
                        region, w, factory, tree_bundle, start_date)
                    cost_w = _evaluate_sequence(
                        factory, region, seq_w, start_date)
                    tree_policy_sweep.append({
                        "tree_w": w,
                        "sequence": seq_w,
                        "sequence_short": [POLICY_SHORT.get(p, p)
                                           for p in seq_w],
                        "cost": cost_w,
                    })
                    print(f"      tree_w={w}: "
                          f"{' -> '.join(POLICY_SHORT.get(p, p) for p in seq_w)} "
                          f"(total={cost_w['total_costs']:.2e})")
                except Exception as e:
                    print(f"      tree_w={w} failed: "
                          f"{type(e).__name__}: {e}")
            valid = [s for s in tree_policy_sweep
                     if s["cost"] is not None
                     and s["cost"].get("total_costs") is not None]
            if valid:
                best = min(valid, key=lambda s: s["cost"]["total_costs"])
                tree_policy_seq = best["sequence"]
                tree_policy_cost = best["cost"]
                tree_policy_weight_used = best["tree_w"]
                print(f"    Best tree weight for {region}: "
                      f"w={tree_policy_weight_used}, sequence: "
                      f"{' -> '.join(POLICY_SHORT.get(p, p) for p in tree_policy_seq)} "
                      f"(total={tree_policy_cost['total_costs']:.2e})")

        all_results[region] = {
            "monthly": monthly,
            "online_sequence": online_seq,
            "actual_sequence": actual_seq,
            "tree_policy_sequence": tree_policy_seq,
            "tree_policy_weight": tree_policy_weight_used,
            "tree_policy_sweep": tree_policy_sweep,
            "online_3mo_cost": online_3mo,
            "real_actual_cost": real_actual,
            "tree_policy_cost": tree_policy_cost,
            "hindsight_optimal": hindsight_optimal,
            "hindsight_all": hindsight_results,
        }
        pd.DataFrame(monthly).to_csv(
            output_dir / f"online_advice_v2_{region}.csv", index=False)

    _plot_online_results_v2(all_results, output_dir)

    with open(output_dir / "online_advisor_v2_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    _print_online_summary_v2(all_results)
    print(f"\n  All outputs saved to: {output_dir}")
    return all_results


REGION_LONG = {"DE": "Germany", "BR": "Brazil", "ES": "Spain", "US-NY": "New York"}


def _plot_online_results_v2(all_results, output_dir):
    """2x2 grouped-bar comparison: online advisor, real actual,
    hindsight optimal (costs as % of quarterly GDP)."""
    regions = list(all_results.keys())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, region in enumerate(regions):
        ax = axes.flat[idx]
        data = all_results[region]
        online = data.get("online_3mo_cost") or {}
        real_actual = data.get("real_actual_cost") or {}
        hindsight = data.get("hindsight_optimal") or {}
        qgdp = TOTAL_GDP[region] / 4.0

        econ = [online.get("economic_costs", 0),
                real_actual.get("economic_costs", 0),
                hindsight.get("economic_costs", 0)]
        human = [online.get("humanitarian_costs", 0),
                 real_actual.get("humanitarian_costs", 0),
                 hindsight.get("humanitarian_costs", 0)]
        total = [e + h for e, h in zip(econ, human)]

        seqs = [data.get("online_sequence", []),
                data.get("actual_sequence", []),
                hindsight.get("policy_vector", []) if hindsight else []]
        seq_labels = [" - ".join(str(POLICY_NUMBER.get(p, p))
                      for p in s) if s else "?" for s in seqs]

        e_pct = [100 * v / qgdp for v in econ]
        h_pct = [100 * v / qgdp for v in human]
        t_pct = [100 * v / qgdp for v in total]

        x = np.arange(3)
        ax.bar(x, e_pct, 0.55, label="Economic", color="#4e79a7",
               alpha=0.85, edgecolor="white")
        ax.bar(x, h_pct, 0.55, bottom=e_pct, label="Humanitarian",
               color="#e15759", alpha=0.85, edgecolor="white")

        for i in range(3):
            ax.text(x[i], t_pct[i] + max(t_pct) * 0.02,
                    f"{t_pct[i]:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(x[i], -max(t_pct) * 0.06, seq_labels[i],
                    ha="center", va="top", fontsize=7, color="gray",
                    style="italic")

        ax.set_ylabel("3-Month Total Cost (% quarterly GDP)", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["Online\nAdvisor", "Real\nActual",
                            "Hindsight\nOptimal"], fontsize=9)
        ax.set_title(REGION_LONG.get(region, region),
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-max(t_pct) * 0.12, max(t_pct) * 1.30)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_dir / "online_advisor_v2_comparison.pdf",
                bbox_inches="tight", dpi=200)
    fig.savefig(output_dir / "online_advisor_v2_comparison.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)


def _print_online_summary_v2(all_results):
    print("\n" + "=" * 90)
    print("  TRUE ONLINE ROLLING-HORIZON ADVISOR (v2 - rank-1 ALS gammas)")
    print("  Actual cost = REAL OBSERVED cost (policy_type='actual')")
    print("  Costs shown as % of quarterly GDP")
    print("=" * 90)
    rows = []
    for region, data in all_results.items():
        on = data.get("online_3mo_cost")
        tp = data.get("tree_policy_cost")
        ac = data.get("real_actual_cost")
        ho = data.get("hindsight_optimal")
        qgdp = TOTAL_GDP[region] / 4.0

        def _seq_nums(seq):
            return "-".join(str(POLICY_NUMBER.get(p, p)) for p in seq)

        on_seq = _seq_nums(data["online_sequence"])
        tp_seq = _seq_nums(data.get("tree_policy_sequence") or [])
        ac_seq = _seq_nums(data["actual_sequence"])
        ho_seq = _seq_nums(ho["policy_vector"] if ho else [])

        on_total = on["total_costs"] if on else None
        tp_total = tp["total_costs"] if tp else None
        ac_total = ac["total_costs"] if ac else None
        ho_total = ho["total_costs"] if ho else None

        def _pct(x):
            return 100 * x / qgdp if x is not None else None

        def _gap(x):
            if x is None or ho_total is None:
                return None
            return 100 * (x - ho_total) / abs(ho_total)

        rows.append({
            "Region": region,
            "Online Seq": on_seq,
            "Actual Seq": ac_seq,
            "HS Seq": ho_seq,
            "Online %GDP": f"{_pct(on_total):.1f}%" if on_total else "N/A",
            "Actual %GDP": f"{_pct(ac_total):.1f}%" if ac_total else "N/A",
            "HS %GDP": f"{_pct(ho_total):.1f}%" if ho_total else "N/A",
            "Online Gap": (f"{_gap(on_total):+.1f}%"
                           if _gap(on_total) is not None else "N/A"),
            "Actual Gap": (f"{_gap(ac_total):+.1f}%"
                           if _gap(ac_total) is not None else "N/A"),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def _revealed_preference_analysis(all_results, factory, output_dir):
    """Reverse-engineer the w* each region's actual policy was targeting,
    then compare hindsight, online, and actual under that revealed w*."""
    output_dir = Path(output_dir)
    w_grid = np.linspace(0, 1, 101)
    results = {}

    print("\n" + "=" * 70)
    print("  REVEALED-PREFERENCE w* ANALYSIS")
    print("=" * 70)

    for region in all_results:
        data = all_results[region]
        hs_all = data.get("hindsight_all", [])
        actual = data.get("real_actual_cost")
        if not hs_all or not actual:
            print(f"  [{region}] skipping: missing data")
            continue

        qgdp = TOTAL_GDP[region] / 4.0
        econ_actual = actual["economic_costs"]
        hum_actual = actual["humanitarian_costs"]

        best_regret_ratio = float("inf")
        revealed_w = 0.5

        for w in w_grid:
            c_actual_w = w * hum_actual + (1 - w) * econ_actual
            c_star_w = min(
                w * r["humanitarian_costs"] + (1 - w) * r["economic_costs"]
                for r in hs_all
            )
            if c_star_w > 0:
                ratio = c_actual_w / c_star_w
            else:
                ratio = float("inf")
            if ratio < best_regret_ratio:
                best_regret_ratio = ratio
                revealed_w = float(w)

        hs_at_w_seq = min(
            hs_all,
            key=lambda r: revealed_w * r["humanitarian_costs"]
                          + (1 - revealed_w) * r["economic_costs"]
        )
        hs_at_w_cost = (revealed_w * hs_at_w_seq["humanitarian_costs"]
                        + (1 - revealed_w) * hs_at_w_seq["economic_costs"])

        on_cost = data.get("online_3mo_cost")
        on_weighted = (revealed_w * on_cost["humanitarian_costs"]
                       + (1 - revealed_w) * on_cost["economic_costs"]
                       ) if on_cost else 0
        actual_weighted = revealed_w * hum_actual + (1 - revealed_w) * econ_actual

        results[region] = {
            "revealed_w": round(revealed_w, 3),
            "regret_ratio": round(best_regret_ratio, 4),
            "hindsight_weighted": hs_at_w_cost,
            "hindsight_seq": hs_at_w_seq.get("policy_vector", []),
            "online_weighted": on_weighted,
            "online_seq": data.get("online_sequence", []),
            "actual_weighted": actual_weighted,
            "actual_seq": data.get("actual_sequence", []),
            "qgdp": qgdp,
        }

        print(f"\n  [{REGION_LONG.get(region, region)}] revealed w* = "
              f"{revealed_w:.3f} (regret ratio = "
              f"{best_regret_ratio:.4f})")
        print(f"    Hindsight  : {hs_at_w_cost/qgdp*100:.1f}% of Q-GDP  "
              f"({'-'.join(str(POLICY_NUMBER.get(p,p)) for p in hs_at_w_seq.get('policy_vector',[]))})")
        print(f"    Online     : {on_weighted/qgdp*100:.1f}% of Q-GDP  "
              f"({'-'.join(str(POLICY_NUMBER.get(p,p)) for p in data.get('online_sequence',[]))})")
        print(f"    Actual     : {actual_weighted/qgdp*100:.1f}% of Q-GDP  "
              f"({'-'.join(str(POLICY_NUMBER.get(p,p)) for p in data.get('actual_sequence',[]))})")

    _plot_revealed_preference(results, output_dir)

    with open(output_dir / "revealed_preference_w.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def _plot_revealed_preference(results, output_dir):
    """Grouped-bar chart: per-region weighted cost under revealed w*."""
    regions = list(results.keys())
    n = len(regions)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        r = results[region]
        qgdp = r["qgdp"]
        labels = ["Hindsight", "Online", "Actual"]
        vals = [
            100 * r["hindsight_weighted"] / qgdp,
            100 * r["online_weighted"] / qgdp,
            100 * r["actual_weighted"] / qgdp,
        ]
        colors = ["#59a14f", "#4e79a7", "#e15759"]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white",
                      width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.02,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_ylabel("Weighted cost (% quarterly GDP)", fontsize=10)
        ax.set_title(f"{REGION_LONG.get(region, region)}\n"
                     f"$w^* = {r['revealed_w']:.2f}$",
                     fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.25)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_dir / "revealed_preference_w.png",
                bbox_inches="tight", dpi=200)
    fig.savefig(output_dir / "revealed_preference_w.pdf",
                bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\n  Revealed-preference figure saved to "
          f"{output_dir / 'revealed_preference_w.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="+", default=REGIONS)
    parser.add_argument("--startdate", default=START_DATE)
    parser.add_argument("--output-dir",
                        default="simulation_results/online_advisor_v2")
    args = parser.parse_args()
    all_results = run_online_rolling_v2(
        regions=args.regions,
        start_date=args.startdate,
        output_dir=args.output_dir,
    )
    factory = Pandemic_Factory()
    factory._initialize_rank1()
    _revealed_preference_analysis(all_results, factory, args.output_dir)


if __name__ == "__main__":
    main()
