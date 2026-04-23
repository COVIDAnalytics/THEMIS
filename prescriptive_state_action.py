"""
State-Action Decision Trees from THEMIS Simulations.

Produces interpretable CART trees that map an observable
pandemic-region state to the cost-optimal next-month NPI choice.

Methodology
-----------
At each decision point t in {1, 2, 3} a policymaker observes:
    (a) pandemic state - cumulative cases, deaths, hospitalisation
        burden as % of population at the start of month t;
    (b) policy history - the average severity of past NPIs and the
        severity of the most recent NPI;
    (c) cost incurred so far - the cumulative economic and humanitarian
        cost as % of monthly GDP that has been spent on the prefix
        policies;
    (d) region characteristics that are also observable ex-ante: GDP
        per capita and population.

For each (region, prefix) combination we enumerate all completions of
the 3-month sequence using the THEMIS forward simulator and identify
the cost-optimal next-month NPI under a weighted total cost
    C(w) = w * humanitarian + (1 - w) * economic.
We do this for several w in [0, 1].  This yields one (state, action)
training row per (region, prefix, w) tuple - 4 regions x 43 prefixes
x 5 weights = 860 rows when pooled.

Design choices
--------------
- Gammas come from the new rank-1 ALS estimator (run_scatter_rank1.py),
  with the calibrated zmean-based fallback for Germany - identical to
  the gamma pipeline now used for the scatter-plot panel.
- The tree is region-blind: it never sees a region indicator, only the
  observable region characteristics above; this keeps the rules
  country-portable.
- We build one tree per w (giving a family of weight-conditional rules)
  and one pooled tree with w as an additional feature.
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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

from pandemic_functions.pandemic import Pandemic_Factory, Pandemic
from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    get_region_gammas_v2,
)
from policy_functions.policy import Policy
from cost_functions.economic_cost.economic_data.economic_params import (
    TOTAL_GDP, TOTAL_LABOR_FORCE,
)
from analyze_gamma_rank import build_gamma_matrix, rank1_imputation
from run_scatter_rank1 import _region_to_matrix_key

FUTURE_POLICIES = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Mass_Gatherings_Authorized_But_Others_Restricted",
    "Restrict_Mass_Gatherings_and_Schools",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]

POLICY_SEVERITY = {p: i for i, p in enumerate(FUTURE_POLICIES)}

POLICY_SHORT = {
    "No_Measure": "None",
    "Restrict_Mass_Gatherings": "MG",
    "Mass_Gatherings_Authorized_But_Others_Restricted": "T+W",
    "Restrict_Mass_Gatherings_and_Schools": "MG+Sch",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others": "MG+T+W",
    "Restrict_Mass_Gatherings_and_Schools_and_Others": "MG+Sch+T+W",
    "Lockdown": "Lock",
}

POLICY_NUMBER = {p: i + 1 for i, p in enumerate(FUTURE_POLICIES)}

REGIONS = ["DE", "BR", "ES", "US-NY"]
START_DATE = "2020-03-15"
GAMMA_WINDOW_END = "2020-06-15"

REGION_POPULATION = {
    "DE": 83_240_000,
    "BR": 212_600_000,
    "ES": 47_350_000,
    "US-NY": 19_540_000,
}

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def compute_rank1_gammas():
    """Rank-1 ALS gammas (no per-region override)."""
    print(f"  Building observed gamma matrix [{START_DATE} -> "
          f"{GAMMA_WINDOW_END}] ...")
    gamma_matrix, obs_mask, region_ids, policy_names = build_gamma_matrix(
        start_date=START_DATE, end_date=GAMMA_WINDOW_END,
    )
    print(f"  Matrix: {gamma_matrix.shape[0]} regions x "
          f"{gamma_matrix.shape[1]} policies, "
          f"{obs_mask.sum()}/{gamma_matrix.size} observed")
    print("  Running rank-1 ALS ...")
    completed, _ = rank1_imputation(gamma_matrix, obs_mask)

    out = {}
    for region in REGIONS:
        key = _region_to_matrix_key(region)
        if key not in region_ids:
            raise KeyError(f"Region '{key}' not in matrix")
        idx = region_ids.index(key)
        out[region] = {p: float(completed[idx, j])
                       for j, p in enumerate(policy_names)}
    return out


def _simulate_sequence(factory, region, policy_vector, gammas):
    """Run THEMIS for one hypothetical sequence and return cost summary."""
    from pandemic_functions.pandemic_params import region_symbol_country_dict
    country, province = region_symbol_country_dict[region]
    csv = (f"pandemic_functions/pandemic_data/"
           f"Cases_{country.replace(' ', '_')}_"
           f"{province.replace(' ', '_')}.csv")
    totalcases = pd.read_csv(csv)
    policy = Policy(policy_type="hypothetical", start_date=START_DATE,
                    policy_vector=policy_vector)
    pandemic = Pandemic(policy, region, factory.delphi_prediction,
                        totalcases, gammas)
    cost = PandemicCost(pandemic)
    econ = float(cost.st_economic_costs)
    human = float(cost.d_costs + cost.h_costs + cost.mh_costs)
    return {
        "policy_vector": tuple(policy_vector),
        "n_months": len(policy_vector),
        "economic_costs": econ,
        "humanitarian_costs": human,
        "d_costs": float(cost.d_costs),
        "h_costs": float(cost.h_costs),
        "mh_costs": float(cost.mh_costs),
        "num_cases": float(cost.num_cases),
        "num_deaths": float(cost.num_deaths),
        "hospitalization_days": float(cost.hospitalization_days),
        "icu_days": float(cost.icu_days),
        "ventilated_days": float(cost.ventilated_days),
    }


def _enumerate_simulations(factory, region, gammas):
    """Run all length-1, length-2 and length-3 hypothetical sequences."""
    print(f"  [{region}] simulating length-1, length-2 and length-3 grids ...")
    by_len = {1: {}, 2: {}, 3: {}}
    for L in (1, 2, 3):
        for pv in itertools.product(FUTURE_POLICIES, repeat=L):
            try:
                r = _simulate_sequence(factory, region, list(pv), gammas)
            except Exception as e:
                print(f"    SKIP len={L} {pv}: {type(e).__name__}: {e}")
                continue
            by_len[L][tuple(pv)] = r
        print(f"    len={L}: {len(by_len[L])} sims")
    return by_len


def _state_at_decision(region, prefix, by_len):
    """Construct the observable state vector at the start of the next
    month given a prefix of past policies."""
    pop = REGION_POPULATION[region]
    gdp_per_capita = TOTAL_GDP[region] / pop
    monthly_gdp = TOTAL_GDP[region] / 12.0

    if len(prefix) == 0:
        return {
            "pop_pct_cum_cases": 0.0,
            "pop_pct_cum_deaths": 0.0,
            "pop_pct_active_hosp": 0.0,
            "month_t": 1,
            "last_severity": 0,
            "mean_prefix_severity": 0,
            "cum_econ_cost_pct_gdp": 0.0,
            "cum_human_cost_pct_gdp": 0.0,
            "cum_total_cost_pct_gdp": 0.0,
        }

    L = len(prefix)
    if tuple(prefix) not in by_len[L]:
        return None
    pref = by_len[L][tuple(prefix)]
    days_in_prefix = 30.0 * L

    def _safe(v):
        return 0.0 if (v is None or not np.isfinite(v)) else float(v)

    econ = _safe(pref["economic_costs"])
    human = _safe(pref["humanitarian_costs"])
    return {
        "pop_pct_cum_cases": 100.0 * _safe(pref["num_cases"]) / pop,
        "pop_pct_cum_deaths": 100.0 * _safe(pref["num_deaths"]) / pop,
        "pop_pct_active_hosp": 100.0 * (_safe(pref["hospitalization_days"])
                                        / days_in_prefix) / pop,
        "month_t": L + 1,
        "last_severity": POLICY_SEVERITY[prefix[-1]],
        "mean_prefix_severity": float(np.mean(
            [POLICY_SEVERITY[p] for p in prefix])),
        "cum_econ_cost_pct_gdp": 100.0 * econ / monthly_gdp,
        "cum_human_cost_pct_gdp": 100.0 * human / monthly_gdp,
        "cum_total_cost_pct_gdp": 100.0 * (econ + human) / monthly_gdp,
    }


def _optimal_next_action(region, prefix, by_len, w):
    """Find the cost-optimal next-month NPI under weight w.

    We enumerate all 6^(3-len(prefix)) full 3-month sequences that begin
    with the prefix, compute weighted cost for each, and return the
    next-month NPI of the minimiser.
    """
    L = len(prefix)
    candidates = []
    for suffix in itertools.product(FUTURE_POLICIES, repeat=3 - L):
        full = tuple(prefix) + suffix
        if full not in by_len[3]:
            continue
        r = by_len[3][full]
        h = r["humanitarian_costs"]
        e = r["economic_costs"]
        if not (np.isfinite(h) and np.isfinite(e)):
            continue
        cost = w * h + (1 - w) * e
        next_npi = full[L]
        candidates.append((cost, next_npi, full))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return {
        "next_npi": candidates[0][1],
        "next_severity": POLICY_SEVERITY[candidates[0][1]],
        "best_cost": candidates[0][0],
        "best_full_sequence": candidates[0][2],
    }


def _build_state_action_dataset(all_sims):
    """Pool (state, action) rows across regions, prefixes and weights."""
    rows = []
    for region, by_len in all_sims.items():
        for L in (0, 1, 2):
            prefixes = [()] if L == 0 else list(
                itertools.product(FUTURE_POLICIES, repeat=L))
            for prefix in prefixes:
                state = _state_at_decision(region, list(prefix), by_len)
                if state is None:
                    continue
                for w in WEIGHTS:
                    opt = _optimal_next_action(region, list(prefix), by_len, w)
                    if opt is None:
                        continue
                    row = dict(state)
                    row["w_humanitarian"] = w
                    row["region"] = region
                    row["prefix_str"] = "|".join(
                        POLICY_SHORT[p] for p in prefix) if prefix else "(start)"
                    row["next_npi"] = opt["next_npi"]
                    row["next_severity"] = opt["next_severity"]
                    row["best_cost"] = opt["best_cost"]
                    row["best_full_sequence"] = "|".join(
                        POLICY_SHORT[p] for p in opt["best_full_sequence"])
                    rows.append(row)
    return pd.DataFrame(rows)


STATE_FEATURES_BASE = [
    "last_severity",
    "mean_prefix_severity",
    "pop_pct_cum_cases",
    "pop_pct_cum_deaths",
    "pop_pct_active_hosp",
    "cum_econ_cost_pct_gdp",
    "cum_human_cost_pct_gdp",
]


def _train_tree(X, y, max_depth=4, min_leaf=4):
    n_classes = y.nunique()
    if n_classes < 2:
        return None
    tree = DecisionTreeClassifier(max_depth=max_depth,
                                  min_samples_leaf=min_leaf,
                                  random_state=42)
    tree.fit(X, y)
    cv = min(5, max(2, len(y) // 5))
    cv_score = cross_val_score(tree, X, y, cv=cv, scoring="accuracy").mean()
    return {
        "tree": tree,
        "feature_names": list(X.columns),
        "classes": tree.classes_.tolist(),
        "cv_accuracy": float(cv_score),
        "n_train": int(len(y)),
    }


def _plot_tree(tree_info, title, out_path):
    """Render a CART using a clean, non-overlapping layout.

    sklearn's default tree renderer often produces overlapping nodes for
    deep, wide trees because matplotlib's text bounding-box estimation
    is conservative. We fix this by sizing the canvas as a function of
    leaf count and tree depth, using a tighter font, and disabling the
    extra impurity / proportion lines that pad each node vertically.
    """
    n_leaves = tree_info["tree"].get_n_leaves()
    depth = tree_info["tree"].get_depth()
    width = max(20.0, 2.4 * n_leaves)
    height = max(8.0, 2.6 * (depth + 1))
    fig, ax = plt.subplots(figsize=(width, height))
    plot_tree(
        tree_info["tree"],
        feature_names=tree_info["feature_names"],
        class_names=[POLICY_SHORT.get(c, str(c))
                     for c in tree_info["classes"]],
        filled=True, rounded=True, fontsize=11, ax=ax,
        impurity=False, proportion=False,
        precision=2,
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax.margins(x=0.02, y=0.02)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    fig.savefig(str(out_path).replace(".png", ".pdf"),
                bbox_inches="tight")
    plt.close(fig)


def _extract_rules(tree_info):
    tree = tree_info["tree"]
    feature_names = tree_info["feature_names"]
    classes = tree_info["classes"]
    rules = []

    def recurse(node, conditions):
        if tree.tree_.feature[node] == -2:
            value = tree.tree_.value[node][0]
            pred = classes[int(np.argmax(value))]
            n = int(tree.tree_.n_node_samples[node])
            total = float(np.sum(value)) or 1.0
            purity = float(np.max(value) / total)
            rules.append({
                "conditions": list(conditions),
                "recommendation": POLICY_SHORT.get(pred, str(pred)),
                "purity": round(purity, 3),
                "n_samples": n,
            })
            return
        fname = feature_names[tree.tree_.feature[node]]
        thr = round(float(tree.tree_.threshold[node]), 3)
        recurse(tree.tree_.children_left[node],
                conditions + [f"{fname} <= {thr}"])
        recurse(tree.tree_.children_right[node],
                conditions + [f"{fname} > {thr}"])
    recurse(0, [])
    return rules


def run_state_action_extraction(output_dir, from_cache=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STATE-ACTION DECISION TREE EXTRACTION")
    print("=" * 70)

    cached = output_dir / "state_action_dataset.csv"
    if from_cache and cached.exists():
        print(f"  Loading cached dataset from {cached}")
        df = pd.read_csv(cached)
        df["next_severity"] = df["next_npi"].map(POLICY_SEVERITY)
        df["prefix_str"] = df["prefix_str"].fillna("(start)")
    else:
        rank1_gammas = compute_rank1_gammas()
        factory = Pandemic_Factory()
        all_sims = {}
        for region in REGIONS:
            all_sims[region] = _enumerate_simulations(factory, region,
                                                       rank1_gammas[region])
        print("\n  Building (state, action) dataset ...")
        df = _build_state_action_dataset(all_sims)
        df.to_csv(output_dir / "state_action_dataset.csv", index=False)
    print(f"  Pooled rows: {len(df)} "
          f"(regions={df['region'].nunique()}, "
          f"weights={sorted(df['w_humanitarian'].unique())})")

    print("\n  Action distribution (next-month NPI) by weight w:")
    for w in WEIGHTS:
        sub = df[df["w_humanitarian"] == w]
        counts = sub["next_npi"].value_counts().to_dict()
        labelled = {POLICY_SHORT[k]: v for k, v in counts.items()}
        print(f"    w={w}: {labelled}")

    rules_records = []
    trees = {}

    print("\n  Training pooled tree with w as a feature "
          "(max_depth=4, min_leaf=20) ...")
    pooled_features = STATE_FEATURES_BASE + ["w_humanitarian"]
    X_pool = df[pooled_features]
    y_pool = df["next_npi"]
    pooled = _train_tree(X_pool, y_pool, max_depth=4, min_leaf=20)
    if pooled is None:
        raise RuntimeError("Pooled tree could not be trained.")
    trees["pooled"] = pooled
    out_path = output_dir / "tree_pooled_with_w.png"
    title = (f"Pooled state-action tree (next-month NPI), w as feature\n"
             f"max_depth = 4, min_leaf = 20, "
             f"5-fold CV acc = {pooled['cv_accuracy']:.3f}, "
             f"n = {pooled['n_train']}, "
             f"depth = {pooled['tree'].get_depth()}")
    _plot_tree(pooled, title, out_path)
    print(f"    pooled: trained, depth={pooled['tree'].get_depth()}, "
          f"CV acc={pooled['cv_accuracy']:.3f}")
    for r in _extract_rules(pooled):
        r["w"] = "pooled"
        rules_records.append(r)

    rules_df = pd.DataFrame(rules_records)
    rules_df.to_csv(output_dir / "state_action_rules.csv", index=False)

    print("\n  All leaves of the pooled tree (sorted by purity desc):")
    ranked = rules_df.sort_values(
        ["purity", "n_samples"], ascending=[False, False])
    for _, row in ranked.iterrows():
        cond = " ; ".join(row["conditions"])
        print(f"    purity={row['purity']:.2f}  n={row['n_samples']:>4d}  "
              f"-> {row['recommendation']:<10s} | {cond}")

    print("\n  Counter-intuitive leaves "
          "(low w prescribes strict NPI, or high w prescribes loose NPI):")

    def _w_floor(conditions):
        floor, ceil = 0.0, 1.0
        for c in conditions:
            if "w_humanitarian" in c and "<=" in c:
                ceil = min(ceil, float(c.split("<=")[-1].strip()))
            elif "w_humanitarian" in c and ">" in c:
                floor = max(floor, float(c.split(">")[-1].strip()))
        return floor, ceil

    LOOSE = {"None", "MG", "T+W"}
    STRICT = {"MG+Sch+T+W", "Lock"}
    interesting_rows = []
    for _, row in rules_df.iterrows():
        floor, ceil = _w_floor(row["conditions"])
        rec = row["recommendation"]
        if floor >= 0.625 and rec in LOOSE:
            interesting_rows.append((floor, ceil, row))
        elif ceil <= 0.375 and rec in STRICT:
            interesting_rows.append((floor, ceil, row))
    if not interesting_rows:
        print("    (no leaf with a counter-intuitive recommendation)")
    for floor, ceil, row in interesting_rows:
        cond = " ; ".join(row["conditions"])
        print(f"    w in ({floor:.3f}, {ceil:.3f}]  purity={row['purity']:.2f}  "
              f"n={row['n_samples']:>4d}  -> {row['recommendation']:<10s} | {cond}")

    with open(output_dir / "state_action_trees.pkl", "wb") as fpkl:
        pickle.dump({
            "trees": trees,
            "state_features_base": STATE_FEATURES_BASE,
            "future_policies": FUTURE_POLICIES,
            "policy_short": POLICY_SHORT,
            "policy_severity": POLICY_SEVERITY,
            "weights": list(WEIGHTS),
        }, fpkl)

    summary = {
        "n_data_rows": int(len(df)),
        "n_regions": int(df["region"].nunique()),
        "n_weights": int(df["w_humanitarian"].nunique()),
        "weights": list(WEIGHTS),
        "trees": {
            (str(k) if k != "pooled" else "pooled"): {
                "cv_accuracy": v["cv_accuracy"],
                "depth": int(v["tree"].get_depth()),
                "n_train": v["n_train"],
                "n_classes": len(v["classes"]),
            }
            for k, v in trees.items()
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot_action_grid(df, output_dir)

    print(f"\n  All outputs saved to {output_dir}")
    print("=" * 70)
    return df, trees, rules_df


def _plot_action_grid(df, output_dir):
    """Heatmap-style summary: optimal action per (region, prefix, w)."""
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(20, 8),
                             sharey=False)
    for ax, region in zip(axes, REGIONS):
        sub = df[df["region"] == region]
        prefixes = sorted(sub["prefix_str"].unique(),
                          key=lambda s: (len(s.split("|")) if s != "(start)"
                                         else 0, s))
        weights = sorted(sub["w_humanitarian"].unique())
        mat = np.zeros((len(prefixes), len(weights)))
        for i, pre in enumerate(prefixes):
            for j, w in enumerate(weights):
                row = sub[(sub["prefix_str"] == pre)
                          & (sub["w_humanitarian"] == w)]
                if len(row) == 0:
                    mat[i, j] = np.nan
                else:
                    mat[i, j] = row["next_severity"].iloc[0]
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=len(FUTURE_POLICIES) - 1)
        ax.set_xticks(range(len(weights)))
        ax.set_xticklabels([f"{w:.2f}" for w in weights], fontsize=9)
        ax.set_yticks(range(len(prefixes)))
        ax.set_yticklabels(prefixes, fontsize=5)
        ax.set_xlabel("Humanitarian weight $w$", fontsize=10)
        ax.set_title(region, fontsize=12, fontweight="bold")
    n_policies = len(FUTURE_POLICIES)
    cbar = fig.colorbar(im, ax=axes, shrink=0.7,
                        ticks=range(n_policies), pad=0.02)
    cbar.ax.set_yticklabels([POLICY_SHORT[FUTURE_POLICIES[i]]
                             for i in range(n_policies)])
    fig.suptitle("Optimal next-month NPI by prefix and weight",
                 fontsize=13, fontweight="bold")
    out = output_dir / "action_grid_by_region.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    fig.savefig(str(out).replace(".png", ".pdf"),
                bbox_inches="tight", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",
                        default="simulation_results/state_action")
    parser.add_argument("--from-cache", action="store_true",
                        help="Load cached state_action_dataset.csv and "
                             "regenerate trees / figures only.")
    args = parser.parse_args()
    run_state_action_extraction(args.output_dir, from_cache=args.from_cache)


if __name__ == "__main__":
    main()
