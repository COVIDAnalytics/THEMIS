"""
Decision-Rule Extraction from THEMIS Simulations.

Trains interpretable decision trees (CART) on the full THEMIS simulation
output to produce human-readable if-then rules that map observable epidemic
state variables to the recommended NPI level at each month of a 3-month
policy sequence.

The key insight is that each month's optimal NPI choice depends on the
epidemic trajectory *at the start of that month*, which is itself a function
of (a) regional characteristics, (b) the current epidemic state, and
(c) the NPI applied in the preceding month.  We therefore build one
decision tree per month position (t=1, t=2, t=3) where features encode
the state observable to a decision-maker at the start of that month.

Usage:
    python prescriptive_decision_rules.py
    python prescriptive_decision_rules.py --regions DE BR ES US-NY --depth 4
"""
import argparse
import itertools
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score

from pandemic_functions.pandemic import Pandemic_Factory, Pandemic
from pandemic_functions.pandemic_cost import PandemicCost
from pandemic_functions.pandemic_params import region_symbol_country_dict
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import (
    get_region_gammas_v2,
    run_delphi_policy_scenario,
)
from policy_functions.policy import Policy
from cost_functions.economic_cost.economic_data.economic_params import (
    TOTAL_GDP, TOTAL_LABOR_FORCE, UNEMPLOYMENT_COST, GDP_IMPACT,
    EMPLOYMENT_IMPACT,
)

FUTURE_POLICIES = [
    "No_Measure",
    "Restrict_Mass_Gatherings",
    "Restrict_Mass_Gatherings_and_Schools",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others",
    "Restrict_Mass_Gatherings_and_Schools_and_Others",
    "Lockdown",
]

POLICY_SEVERITY = {p: i for i, p in enumerate(FUTURE_POLICIES)}

POLICY_SHORT = {
    "No_Measure": "None",
    "Restrict_Mass_Gatherings": "MG",
    "Restrict_Mass_Gatherings_and_Schools": "MG+Sch",
    "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others": "Sch+MG+Oth",
    "Restrict_Mass_Gatherings_and_Schools_and_Others": "MG+Sch+Oth",
    "Lockdown": "Lock",
}

REGIONS = ["DE", "BR", "ES", "US-NY"]


def _simulate_all_policies(factory, region, start_date, policy_length):
    """Run the full policy grid and return a DataFrame with costs."""
    base_gammas, _, _ = get_region_gammas_v2(region)
    country, province = region_symbol_country_dict[region]
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")
    totalcases = pd.read_csv(
        f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
    )

    scenarios = list(itertools.product(FUTURE_POLICIES, repeat=policy_length))
    rows = []
    for pv in scenarios:
        pv_list = list(pv)
        try:
            policy = Policy(policy_type="hypothetical", start_date=start_date,
                            policy_vector=pv_list)
            pandemic = Pandemic(policy, region, factory.delphi_prediction,
                                totalcases, base_gammas)
            cost = PandemicCost(pandemic)
            econ = cost.st_economic_costs
            humanitarian = cost.d_costs + cost.h_costs + cost.mh_costs
            rows.append({
                "region": region,
                "policy_vector": pv_list,
                "month1": pv_list[0],
                "month2": pv_list[1],
                "month3": pv_list[2],
                "economic_costs": econ,
                "humanitarian_costs": humanitarian,
                "total_costs": econ + humanitarian,
                "num_cases": cost.num_cases,
                "num_deaths": cost.num_deaths,
                "d_costs": cost.d_costs,
                "h_costs": cost.h_costs,
                "mh_costs": cost.mh_costs,
                "hospitalization_days": cost.hospitalization_days,
                "icu_days": cost.icu_days,
            })
        except Exception as e:
            print(f"  SKIP {pv_list}: {e}")
            continue
    return pd.DataFrame(rows), base_gammas


def _build_features(df, region):
    """Build decision-tree features for each observation (policy sequence)."""
    gdp = TOTAL_GDP.get(region, 1e12)
    labor = TOTAL_LABOR_FORCE.get(region, 1e7)

    records = []
    for _, row in df.iterrows():
        pv = row["policy_vector"]
        base = {
            "region_gdp_per_capita": gdp / labor,
            "month1_severity": POLICY_SEVERITY[pv[0]],
            "month2_severity": POLICY_SEVERITY[pv[1]],
            "month3_severity": POLICY_SEVERITY[pv[2]],
            "avg_severity": np.mean([POLICY_SEVERITY[p] for p in pv]),
            "max_severity": max(POLICY_SEVERITY[p] for p in pv),
            "severity_trend": POLICY_SEVERITY[pv[2]] - POLICY_SEVERITY[pv[0]],
            "starts_strict": int(POLICY_SEVERITY[pv[0]] >= 4),
            "ends_strict": int(POLICY_SEVERITY[pv[2]] >= 4),
            "escalates": int(POLICY_SEVERITY[pv[1]] > POLICY_SEVERITY[pv[0]] or
                             POLICY_SEVERITY[pv[2]] > POLICY_SEVERITY[pv[1]]),
            "de_escalates": int(POLICY_SEVERITY[pv[1]] < POLICY_SEVERITY[pv[0]] or
                                POLICY_SEVERITY[pv[2]] < POLICY_SEVERITY[pv[1]]),
        }
        for col in ["total_costs", "economic_costs", "humanitarian_costs",
                     "num_cases", "num_deaths"]:
            base[col] = row[col]
        records.append(base)
    return pd.DataFrame(records)


def _label_policies(df, top_frac=0.15):
    """Classify each policy sequence as optimal / near-optimal / dominated."""
    total = df["total_costs"]
    cutoff_good = total.quantile(top_frac)
    cutoff_mid = total.quantile(0.5)
    labels = pd.Series("dominated", index=df.index)
    labels[total <= cutoff_mid] = "acceptable"
    labels[total <= cutoff_good] = "near-optimal"
    labels[total == total.min()] = "optimal"
    return labels


def _train_monthly_trees(df, region, max_depth=4):
    """Train one tree per month position predicting the optimal NPI choice."""
    trees = {}
    for month_idx in range(1, 4):
        month_col = f"month{month_idx}"
        top_policies = df.nsmallest(max(1, len(df) // 6), "total_costs")

        prev_cols = [f"month{m}_severity" for m in range(1, month_idx)]
        feature_cols = ["region_gdp_per_capita"] + prev_cols

        X = top_policies[feature_cols].copy() if feature_cols else pd.DataFrame(
            index=top_policies.index)

        X["prev_total_severity"] = sum(
            top_policies[f"month{m}_severity"] for m in range(1, month_idx)
        ) if month_idx > 1 else 0

        y = top_policies[month_col].map(POLICY_SEVERITY)

        if len(y.unique()) < 2:
            trees[month_idx] = {"status": "unanimous",
                                "policy": top_policies[month_col].mode().iloc[0]}
            continue

        tree = DecisionTreeClassifier(max_depth=min(max_depth, 3),
                                      min_samples_leaf=max(2, len(y) // 10),
                                      random_state=42)
        feature_names = list(X.columns)
        tree.fit(X, y)

        cv_acc = cross_val_score(tree, X, y, cv=min(5, len(y)), scoring="accuracy")

        trees[month_idx] = {
            "status": "trained",
            "tree": tree,
            "feature_names": feature_names,
            "cv_accuracy": float(cv_acc.mean()),
            "classes": [FUTURE_POLICIES[int(c)] for c in tree.classes_],
        }
    return trees


def _train_overall_tree(feat_df, df, max_depth=4):
    """Train a single tree predicting optimal vs dominated from sequence features."""
    labels = _label_policies(df)
    target = (labels.isin(["optimal", "near-optimal"])).astype(int)

    feature_cols = ["avg_severity", "max_severity", "severity_trend",
                    "starts_strict", "ends_strict", "escalates", "de_escalates",
                    "month1_severity", "month2_severity", "month3_severity"]
    X = feat_df[feature_cols]
    y = target

    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=5,
                                  random_state=42)
    tree.fit(X, y)
    cv_acc = cross_val_score(tree, X, y, cv=5, scoring="accuracy")

    return tree, feature_cols, float(cv_acc.mean())


def _extract_rules(tree_clf, feature_names, class_names=None):
    """Extract human-readable if-then rules from a fitted decision tree."""
    tree_ = tree_clf.tree_
    rules = []

    def recurse(node, conditions):
        if tree_.feature[node] == -2:
            pred_class = int(np.argmax(tree_.value[node]))
            n_samples = int(tree_.n_node_samples[node])
            confidence = float(tree_.value[node][0][pred_class] / n_samples)
            label = class_names[pred_class] if class_names else str(pred_class)
            rules.append({
                "conditions": list(conditions),
                "recommendation": label,
                "confidence": round(confidence, 3),
                "n_samples": n_samples,
            })
            return
        fname = feature_names[tree_.feature[node]]
        threshold = round(float(tree_.threshold[node]), 3)
        recurse(tree_.children_left[node],
                conditions + [f"{fname} <= {threshold}"])
        recurse(tree_.children_right[node],
                conditions + [f"{fname} > {threshold}"])

    recurse(0, [])
    return rules


def run_decision_rule_extraction(regions, start_date, policy_length, max_depth,
                                 output_dir):
    """Main entry point: simulate, train trees, extract rules for all regions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factory = Pandemic_Factory()
    all_results = {}
    stored_trees = {}
    all_sim_dfs = {}

    for region in regions:
        print(f"\n{'='*60}")
        print(f"  Region: {region}")
        print(f"{'='*60}")

        print("  Simulating all policy sequences ...")
        df, base_gammas = _simulate_all_policies(factory, region, start_date,
                                                  policy_length)
        print(f"  {len(df)} simulations completed.")

        if len(df) == 0:
            continue

        feat_df = _build_features(df, region)

        optimal_idx = df["total_costs"].idxmin()
        optimal_policy = df.loc[optimal_idx, "policy_vector"]
        print(f"  Optimal policy: {[POLICY_SHORT[p] for p in optimal_policy]}")
        print(f"  Optimal total cost: {df.loc[optimal_idx, 'total_costs']:.2e}")

        print("  Training overall decision tree ...")
        overall_tree, overall_features, overall_cv = _train_overall_tree(
            feat_df, df, max_depth)
        overall_rules = _extract_rules(
            overall_tree, overall_features,
            class_names=["Dominated", "Near-Optimal"])
        print(f"  Overall tree CV accuracy: {overall_cv:.3f}")
        print(f"  Extracted {len(overall_rules)} rules")

        severity_labels = {v: POLICY_SHORT[k] for k, v in POLICY_SEVERITY.items()}

        actionable_rules = []
        for rule in overall_rules:
            if rule["recommendation"] == "Near-Optimal":
                readable = _make_readable_rule(rule, severity_labels)
                actionable_rules.append(readable)
                print(f"    RULE: {readable}")

        stored_trees[region] = (overall_tree, overall_features)

        fig, ax = plt.subplots(figsize=(12, 5))
        plot_tree(overall_tree, ax=ax, feature_names=overall_features,
                  class_names=["Dominated", "Near-Optimal"],
                  filled=True, rounded=True, fontsize=10, impurity=False)
        ax.set_title(f"Decision Tree: {region}", fontsize=13)
        plt.tight_layout()
        fig.savefig(output_dir / f"decision_tree_{region}.pdf",
                    bbox_inches="tight", dpi=200)
        fig.savefig(output_dir / f"decision_tree_{region}.png",
                    bbox_inches="tight", dpi=200)
        plt.close(fig)

        tree_text = export_text(overall_tree, feature_names=overall_features)
        (output_dir / f"decision_tree_{region}.txt").write_text(tree_text)

        all_results[region] = {
            "optimal_policy": optimal_policy,
            "optimal_total_cost": float(df.loc[optimal_idx, "total_costs"]),
            "n_simulations": len(df),
            "overall_cv_accuracy": overall_cv,
            "rules": overall_rules,
            "actionable_rules": actionable_rules,
            "base_gammas": {k: round(v, 4) for k, v in base_gammas.items()},
        }

        df.to_csv(output_dir / f"simulations_{region}.csv", index=False)
        all_sim_dfs[region] = df

    if len(stored_trees) >= 4:
        plot_regions = list(stored_trees.keys())[:4]
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        for ax, reg in zip(axes.flat, plot_regions):
            tree_obj, feat_names = stored_trees[reg]
            plot_tree(tree_obj, ax=ax, feature_names=feat_names,
                      class_names=["Dominated", "Near-Optimal"],
                      filled=True, rounded=True, fontsize=8, impurity=False)
            ax.set_title(f"{reg}", fontsize=12, fontweight="bold")
        plt.tight_layout(pad=1.5)
        fig.savefig(output_dir / "decision_trees_combined.pdf",
                    bbox_inches="tight", dpi=200)
        fig.savefig(output_dir / "decision_trees_combined.png",
                    bbox_inches="tight", dpi=200)
        plt.close(fig)

    _build_cross_regional_summary(all_results, output_dir)

    if len(all_sim_dfs) >= 2:
        print("\n\n" + "=" * 60)
        print("  POOLED CROSS-REGIONAL TREE")
        print("=" * 60)
        (pooled_tree, pooled_features, pooled_cv, pooled_rules, pooled_df,
         pooled_importances) = _train_pooled_tree(all_sim_dfs, output_dir,
                                                    max_depth)
        all_results["_pooled"] = {
            "cv_accuracy": pooled_cv,
            "rules": pooled_rules,
            "feature_names": pooled_features,
            "feature_importances": {k: float(v) for k, v in
                                     pooled_importances.items()},
        }

        print("\n\n" + "=" * 60)
        print("  CROSS-COUNTRY ESCALATION RULE")
        print("=" * 60)
        escalation_summary, escalation_rule = _compute_cross_country_escalation(
            pooled_df, output_dir)
        all_results["_cross_country_escalation"] = {
            "summary": escalation_summary,
            "rule": escalation_rule,
        }

        insights = _compute_novel_insights(all_sim_dfs, output_dir)
        all_results["_insights"] = insights

    with open(output_dir / "decision_rules_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  All outputs saved to: {output_dir}")
    return all_results


def _make_readable_rule(rule, severity_labels):
    """Convert a raw rule into a human-readable policy recommendation."""
    conditions = []
    for cond in rule["conditions"]:
        readable = cond
        for sev_val, sev_name in severity_labels.items():
            if str(float(sev_val)) in cond or str(sev_val) in cond:
                readable = cond + f" ({sev_name})"
                break
        conditions.append(readable)
    cond_str = " AND ".join(conditions)
    return (f"IF {cond_str} THEN recommend Near-Optimal sequence "
            f"(confidence={rule['confidence']:.0%}, n={rule['n_samples']})")


def _build_cross_regional_summary(all_results, output_dir):
    """Summarize which patterns are universal vs region-specific."""
    summary_rows = []
    for region, data in all_results.items():
        opt = data["optimal_policy"]
        summary_rows.append({
            "region": region,
            "optimal_month1": POLICY_SHORT.get(opt[0], opt[0]),
            "optimal_month2": POLICY_SHORT.get(opt[1], opt[1]),
            "optimal_month3": POLICY_SHORT.get(opt[2], opt[2]),
            "optimal_total_cost": data["optimal_total_cost"],
            "cv_accuracy": data["overall_cv_accuracy"],
            "n_rules": len(data["actionable_rules"]),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "cross_regional_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("  CROSS-REGIONAL SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    all_optimal_sev = []
    for region, data in all_results.items():
        opt = data["optimal_policy"]
        all_optimal_sev.append([POLICY_SEVERITY[p] for p in opt])

    if all_optimal_sev:
        avg_sev = np.mean(all_optimal_sev, axis=0)
        print(f"\n  Average optimal severity by month: "
              f"{[round(s, 2) for s in avg_sev]}")
        if avg_sev[0] > avg_sev[2]:
            print("  UNIVERSAL PATTERN: Optimal policies tend to de-escalate over time")
        elif avg_sev[0] < avg_sev[2]:
            print("  UNIVERSAL PATTERN: Optimal policies tend to escalate over time")
        else:
            print("  PATTERN: No consistent escalation/de-escalation trend")


def _train_pooled_tree(all_sim_dfs, output_dir, max_depth=3):
    """
    Train a single decision tree pooling all regions and policy choices.

    Features include region-policy *interaction* characteristics so the
    tree can discover rules of the form
        "IF month-1-NPI's GDP impact in region R is below X
            AND humanitarian share is above Y
         THEN near-optimal."
    These rules are policy-attribute-based and therefore portable to any
    new region for which the attributes can be computed.
    """
    frames = []
    for region, df in all_sim_dfs.items():
        dfc = df.copy()
        gdp = TOTAL_GDP.get(region, 1e12)
        labor = TOTAL_LABOR_FORCE.get(region, 1e7)
        lock_gdp_pct = abs(GDP_IMPACT.get(region, {}).get("Lockdown", -5))
        none_gdp_pct = abs(GDP_IMPACT.get(region, {}).get("No_Measure", 0))
        gdp_dispersion = lock_gdp_pct - none_gdp_pct

        dfc["region"] = region
        dfc["gdp_per_worker"] = gdp / labor
        dfc["lockdown_gdp_impact_pct"] = lock_gdp_pct
        dfc["gdp_impact_range_pct"] = gdp_dispersion
        dfc["humanitarian_share"] = (dfc["humanitarian_costs"] /
                                     dfc["total_costs"].clip(lower=1))
        dfc["econ_pct_gdp"] = 100 * dfc["economic_costs"] / gdp
        dfc["death_cost_share"] = dfc["d_costs"] / dfc["total_costs"].clip(lower=1)

        for idx, row in dfc.iterrows():
            pv = row["policy_vector"]
            m1, m2, m3 = pv[0], pv[1], pv[2]
            sev1, sev2, sev3 = (POLICY_SEVERITY[m1],
                                 POLICY_SEVERITY[m2],
                                 POLICY_SEVERITY[m3])

            dfc.loc[idx, "month1_severity"] = sev1
            dfc.loc[idx, "month2_severity"] = sev2
            dfc.loc[idx, "month3_severity"] = sev3
            dfc.loc[idx, "avg_severity"] = (sev1 + sev2 + sev3) / 3
            dfc.loc[idx, "severity_trend"] = sev3 - sev1
            dfc.loc[idx, "starts_strict"] = int(sev1 >= 4)
            dfc.loc[idx, "de_escalates"] = int(sev2 < sev1 or sev3 < sev2)

            m1_gdp = abs(GDP_IMPACT.get(region, {}).get(m1, 0))
            m2_gdp = abs(GDP_IMPACT.get(region, {}).get(m2, 0))
            m3_gdp = abs(GDP_IMPACT.get(region, {}).get(m3, 0))
            dfc.loc[idx, "m1_gdp_impact_pct"] = m1_gdp
            dfc.loc[idx, "m2_gdp_impact_pct"] = m2_gdp
            dfc.loc[idx, "m3_gdp_impact_pct"] = m3_gdp
            dfc.loc[idx, "total_gdp_impact_pct"] = m1_gdp + m2_gdp + m3_gdp
            dfc.loc[idx, "max_month_gdp_impact_pct"] = max(m1_gdp, m2_gdp, m3_gdp)

            m1_emp = EMPLOYMENT_IMPACT.get(region, {}).get(m1, 0)
            m2_emp = EMPLOYMENT_IMPACT.get(region, {}).get(m2, 0)
            m3_emp = EMPLOYMENT_IMPACT.get(region, {}).get(m3, 0)
            try:
                m1_emp = float(m1_emp)
            except (TypeError, ValueError):
                m1_emp = 0.0
            try:
                m2_emp = float(m2_emp)
            except (TypeError, ValueError):
                m2_emp = 0.0
            try:
                m3_emp = float(m3_emp)
            except (TypeError, ValueError):
                m3_emp = 0.0
            dfc.loc[idx, "m1_employment_impact_pct"] = m1_emp
            dfc.loc[idx, "total_employment_impact_pct"] = m1_emp + m2_emp + m3_emp

            dfc.loc[idx, "m1_intensity_ratio"] = (m1_gdp / lock_gdp_pct
                                                   if lock_gdp_pct > 0 else 0)
            dfc.loc[idx, "avg_intensity_ratio"] = (
                (m1_gdp + m2_gdp + m3_gdp) / (3 * lock_gdp_pct)
                if lock_gdp_pct > 0 else 0
            )

        labels = _label_policies(dfc)
        dfc["target"] = (labels.isin(["optimal", "near-optimal"])).astype(int)
        frames.append(dfc)

    pooled = pd.concat(frames, ignore_index=True)

    feature_cols = [
        "m1_gdp_impact_pct", "total_gdp_impact_pct", "max_month_gdp_impact_pct",
        "m1_employment_impact_pct", "total_employment_impact_pct",
        "m1_intensity_ratio", "avg_intensity_ratio",
        "lockdown_gdp_impact_pct", "gdp_impact_range_pct",
        "month1_severity", "month2_severity", "month3_severity",
        "avg_severity", "severity_trend", "starts_strict", "de_escalates",
    ]
    X = pooled[feature_cols].astype(float)
    y = pooled["target"]

    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=10,
                                  random_state=42)
    tree.fit(X, y)
    cv_acc = cross_val_score(tree, X, y, cv=5, scoring="accuracy")
    print(f"\n  Pooled tree CV accuracy: {cv_acc.mean():.3f}")

    rules = _extract_rules(tree, feature_cols, ["Dominated", "Near-Optimal"])
    print(f"  Extracted {len(rules)} rules; near-optimal leaves:")
    near_opt = [r for r in rules if r["recommendation"] == "Near-Optimal"]
    for r in near_opt:
        conds = " AND ".join(r["conditions"])
        print(f"    IF {conds} => Near-Optimal "
              f"(purity={r['confidence']:.0%}, n={r['n_samples']})")

    importances = pd.Series(tree.feature_importances_,
                             index=feature_cols).sort_values(ascending=False)
    print("\n  Feature importances (pooled tree):")
    for name, imp in importances.items():
        if imp > 0.005:
            print(f"    {name:<35}: {imp:.3f}")

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree, ax=ax, feature_names=feature_cols,
              class_names=["Dominated", "Near-Optimal"],
              filled=True, rounded=True, fontsize=9, impurity=False)
    ax.set_title("Pooled Decision Tree: Region-Policy Interaction Features "
                 "(All 4 Regions, n=864)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "decision_tree_pooled.pdf",
                bbox_inches="tight", dpi=200)
    fig.savefig(output_dir / "decision_tree_pooled.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)

    return (tree, feature_cols, float(cv_acc.mean()), rules, pooled,
            importances)


def _compute_cross_country_escalation(pooled_df, output_dir):
    """
    Cross-country escalation/de-escalation rule.

    Pool all (region, sequence) pairs and ask: as a function of the
    month-1 NPI's intensity ratio (its GDP impact relative to lockdown
    in that region), what fraction of near-optimal sequences escalate
    vs. de-escalate by month 3?  This produces a single, country-agnostic
    table that any decision-maker can use.
    """
    near = pooled_df[pooled_df["target"] == 1].copy()
    bins = [-0.01, 0.10, 0.30, 0.55, 0.80, 1.01]
    labels = ["≤10% (near-None)", "10–30% (light)", "30–55% (moderate)",
              "55–80% (strict)", "80–100% (Lockdown-class)"]
    near["m1_intensity_bin"] = pd.cut(near["m1_intensity_ratio"], bins=bins,
                                       labels=labels, include_lowest=True)

    # Tally near-optimal sequences in each bin and compute the average
    # severity trend (positive = escalates, negative = de-escalates).
    summary_rows = []
    all_rows = []
    for label in labels:
        sub = near[near["m1_intensity_bin"] == label]
        if len(sub) == 0:
            continue
        regions_with_near_opt = sub["region"].unique().tolist()
        avg_trend = sub["severity_trend"].mean()
        avg_m2_intensity = sub["m1_intensity_ratio"].apply(
            lambda x: x  # placeholder; real m2 intensity below
        )
        # Compute average m2/m3 severity to characterize the optimal path
        avg_m1_sev = sub["month1_severity"].mean()
        avg_m2_sev = sub["month2_severity"].mean()
        avg_m3_sev = sub["month3_severity"].mean()
        de_escalate_pct = (sub["de_escalates"] == 1).mean() * 100
        summary_rows.append({
            "month1_intensity_bin": label,
            "n_near_optimal": len(sub),
            "n_regions_supporting": len(regions_with_near_opt),
            "regions_supporting": ", ".join(sorted(regions_with_near_opt)),
            "avg_m1_severity": float(avg_m1_sev),
            "avg_m2_severity": float(avg_m2_sev),
            "avg_m3_severity": float(avg_m3_sev),
            "avg_severity_trend": float(avg_trend),
            "pct_that_de_escalate": float(de_escalate_pct),
        })
        all_rows.append(sub)

    print("\n  CROSS-COUNTRY ESCALATION TABLE (pooled near-optimal sequences)")
    print("  " + "-" * 70)
    print(f"  {'Month-1 intensity':<25} {'n':>4} {'#reg':>5} "
          f"{'avg m1':>7} {'avg m2':>7} {'avg m3':>7} "
          f"{'%trend<0':>9} {'%any-deesc':>11}")
    for r in summary_rows:
        sub = near[near["m1_intensity_bin"] == r["month1_intensity_bin"]]
        pct_trend_neg = 100 * (sub["severity_trend"] < 0).mean()
        print(f"  {r['month1_intensity_bin']:<25} {r['n_near_optimal']:>4} "
              f"{r['n_regions_supporting']:>5} "
              f"{r['avg_m1_severity']:>7.2f} {r['avg_m2_severity']:>7.2f} "
              f"{r['avg_m3_severity']:>7.2f} "
              f"{pct_trend_neg:>8.1f}% "
              f"{r['pct_that_de_escalate']:>10.1f}%")

    # Country-agnostic escalation rule: when is escalation (s_3 > s_1)
    # near-optimal vs. when is de-escalation (s_3 < s_1) near-optimal,
    # purely as a function of policy attributes?
    rule_rows = []
    for label in labels:
        sub = near[near["m1_intensity_bin"] == label]
        n = len(sub)
        if n == 0:
            rule_rows.append({
                "month1_intensity_bin": label, "n_near_optimal": 0,
                "pct_de_escalate": 0.0, "pct_constant": 0.0,
                "pct_escalate": 0.0,
            })
            continue
        n_de_escalate = (sub["severity_trend"] < 0).sum()
        n_constant = (sub["severity_trend"] == 0).sum()
        n_escalate = (sub["severity_trend"] > 0).sum()
        rule_rows.append({
            "month1_intensity_bin": label,
            "n_near_optimal": int(n),
            "pct_de_escalate": 100 * n_de_escalate / n,
            "pct_constant": 100 * n_constant / n,
            "pct_escalate": 100 * n_escalate / n,
        })
    rule_df = pd.DataFrame(rule_rows)
    rule_df.to_csv(output_dir / "cross_country_escalation_rule.csv",
                   index=False)

    # Plot: stacked bar of escalation pattern by m1 intensity bin,
    # plus the count of near-optimal sequences per bin
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5),
                                    gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(rule_df))
    bar_labels = ["LightAt", "LightBe", "LightAb", "Mid", "StrictBe", "Lock"]
    short_labels = ["<10%", "10-30%", "30-55%", "55-80%", "80-100%"]
    rule_df_for_plot = rule_df.copy()

    ax1.bar(x, rule_df_for_plot["pct_de_escalate"], 0.65,
            label="De-escalate (s_3 < s_1)",
            color="#59a14f", edgecolor="white")
    ax1.bar(x, rule_df_for_plot["pct_constant"], 0.65,
            bottom=rule_df_for_plot["pct_de_escalate"],
            label="Constant (s_3 = s_1)",
            color="#f28e2b", edgecolor="white")
    ax1.bar(x, rule_df_for_plot["pct_escalate"], 0.65,
            bottom=rule_df_for_plot["pct_de_escalate"] +
                   rule_df_for_plot["pct_constant"],
            label="Escalate (s_3 > s_1)",
            color="#e15759", edgecolor="white")

    for i, n in enumerate(rule_df_for_plot["n_near_optimal"]):
        if n > 0:
            ax1.text(i, 102, f"n={int(n)}", ha="center", fontsize=9,
                     color="black")
        else:
            ax1.text(i, 50, "no near-\noptimal", ha="center", fontsize=8,
                     color="gray", style="italic")

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=10)
    ax1.set_ylabel("% of near-optimal sequences", fontsize=11)
    ax1.set_xlabel("Month-1 NPI intensity (% of Lockdown's GDP impact)",
                    fontsize=11)
    ax1.set_title("Trajectory of near-optimal sequences",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
                fontsize=9)
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", alpha=0.3)

    n_per_bin = rule_df_for_plot["n_near_optimal"].values
    region_support = []
    for label in labels:
        sub = near[near["m1_intensity_bin"] == label]
        region_support.append(sub["region"].nunique())
    ax2.bar(x, region_support, 0.65, color="#4e79a7", edgecolor="white",
             alpha=0.85)
    for i, (n, r) in enumerate(zip(n_per_bin, region_support)):
        if n > 0:
            ax2.text(i, r + 0.05, f"{int(r)}/4", ha="center", fontsize=9,
                      fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=10)
    ax2.set_ylabel("# regions with near-optimal sequences", fontsize=11)
    ax2.set_xlabel("Month-1 NPI intensity (% of Lockdown's GDP impact)",
                    fontsize=11)
    ax2.set_title("Cross-country support: how many of the 4 regions admit "
                  "near-optimal\nsequences in this intensity bin",
                  fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 5)
    ax2.set_yticks(range(5))
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Cross-country escalation rule (pooled across all 4 regions)",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "cross_country_escalation.pdf",
                bbox_inches="tight", dpi=200)
    fig.savefig(output_dir / "cross_country_escalation.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)

    return summary_rows, rule_rows


def _compute_novel_insights(all_sim_dfs, output_dir):
    """Compute non-obvious insights from the THEMIS simulation data."""
    print("\n" + "=" * 70)
    print("  NOVEL PRESCRIPTIVE INSIGHTS")
    print("=" * 70)
    insights = {}

    # 1. Marginal cost of each severity step (month 1 only, avg over months 2-3)
    print("\n  1. MARGINAL COST OF SEVERITY ESCALATION (month-1 NPI)")
    print("  " + "-" * 60)
    marginal_data = []
    for region, df in all_sim_dfs.items():
        gdp = TOTAL_GDP.get(region, 1e12)
        for sev in range(6):
            npi = SEVERITY_TO_POLICY.get(sev, FUTURE_POLICIES[0])
            subset = df[df["month1"] == npi]
            if len(subset) == 0:
                continue
            avg_econ = subset["economic_costs"].mean()
            avg_human = subset["humanitarian_costs"].mean()
            avg_total = subset["total_costs"].mean()
            marginal_data.append({
                "region": region, "severity": sev,
                "npi": POLICY_SHORT[npi],
                "avg_econ": avg_econ, "avg_human": avg_human,
                "avg_total": avg_total,
                "econ_pct_gdp": 100 * avg_econ / gdp,
            })
    mdf = pd.DataFrame(marginal_data)

    for region in all_sim_dfs:
        rdf = mdf[mdf["region"] == region].sort_values("severity")
        if len(rdf) < 2:
            continue
        print(f"\n    {region}:")
        prev = None
        for _, row in rdf.iterrows():
            if prev is not None:
                d_econ = row["avg_econ"] - prev["avg_econ"]
                d_human = row["avg_human"] - prev["avg_human"]
                d_total = row["avg_total"] - prev["avg_total"]
                print(f"      {prev['npi']:>12} -> {row['npi']:<12}: "
                      f"d_econ={d_econ/1e9:+.1f}B, d_human={d_human/1e9:+.1f}B, "
                      f"d_total={d_total/1e9:+.1f}B")
            prev = row

    # 2. Month-2 sensitivity: how much does month-2 matter GIVEN month-1?
    print("\n\n  2. MONTH-2 SENSITIVITY: Does month-2 choice matter after strict month-1?")
    print("  " + "-" * 60)
    sensitivity_data = []
    for region, df in all_sim_dfs.items():
        for m1_npi in FUTURE_POLICIES:
            m1_subset = df[df["month1"] == m1_npi]
            if len(m1_subset) == 0:
                continue
            cost_range = m1_subset["total_costs"].max() - m1_subset["total_costs"].min()
            cost_cv = m1_subset["total_costs"].std() / m1_subset["total_costs"].mean()
            sensitivity_data.append({
                "region": region,
                "month1_npi": POLICY_SHORT[m1_npi],
                "month1_severity": POLICY_SEVERITY[m1_npi],
                "cost_range": cost_range,
                "cost_cv": cost_cv,
                "n_scenarios": len(m1_subset),
            })
    sdf = pd.DataFrame(sensitivity_data)
    for region in all_sim_dfs:
        rdf = sdf[sdf["region"] == region].sort_values("month1_severity")
        print(f"\n    {region}:")
        for _, row in rdf.iterrows():
            print(f"      Month-1={row['month1_npi']:>12}: "
                  f"cost CV={row['cost_cv']:.3f}, "
                  f"range={row['cost_range']/1e9:.1f}B")

    # 3. The lockdown crossover: at what death rate does lockdown become cheaper?
    print("\n\n  3. LOCKDOWN CROSSOVER ANALYSIS")
    print("  " + "-" * 60)
    crossover_data = []
    for region, df in all_sim_dfs.items():
        none_best = df[df["month1"] == "No_Measure"]["total_costs"].min()
        lock_best = df[df["month1"] == "Lockdown"]["total_costs"].min()
        none_deaths = df[df["month1"] == "No_Measure"]["num_deaths"].mean()
        lock_deaths = df[df["month1"] == "Lockdown"]["num_deaths"].mean()
        none_econ = df[df["month1"] == "No_Measure"]["economic_costs"].mean()
        lock_econ = df[df["month1"] == "Lockdown"]["economic_costs"].mean()
        econ_premium = lock_econ - none_econ
        lives_saved = none_deaths - lock_deaths
        cost_per_life = econ_premium / max(lives_saved, 1)
        crossover_data.append({
            "region": region,
            "no_measure_best_total": none_best,
            "lockdown_best_total": lock_best,
            "lockdown_cheaper": lock_best < none_best,
            "econ_premium_of_lock": econ_premium,
            "lives_saved_by_lock": lives_saved,
            "cost_per_life_saved": cost_per_life,
        })
        print(f"    {region}: Lock cheaper? {lock_best < none_best}, "
              f"lives saved={lives_saved:.0f}, "
              f"econ premium=${econ_premium/1e9:.1f}B, "
              f"cost/life=${cost_per_life/1e6:.2f}M")

    # 4. The "diminishing returns" of month-2 strictness
    print("\n\n  4. DIMINISHING RETURNS: Month-2 strictness after Lockdown month-1")
    print("  " + "-" * 60)
    for region, df in all_sim_dfs.items():
        lock_m1 = df[df["month1"] == "Lockdown"]
        if len(lock_m1) == 0:
            continue
        m2_costs = lock_m1.groupby("month2").agg({
            "total_costs": "mean",
            "economic_costs": "mean",
            "humanitarian_costs": "mean",
            "num_deaths": "mean",
        }).reset_index()
        m2_costs["month2_short"] = m2_costs["month2"].map(POLICY_SHORT)
        m2_costs["severity"] = m2_costs["month2"].map(POLICY_SEVERITY)
        m2_costs = m2_costs.sort_values("severity")
        print(f"\n    {region} (month-1 = Lock):")
        for _, row in m2_costs.iterrows():
            print(f"      M2={row['month2_short']:>12}: "
                  f"total={row['total_costs']/1e9:.1f}B, "
                  f"econ={row['economic_costs']/1e9:.1f}B, "
                  f"deaths={row['num_deaths']:.0f}")

    # 5. Humanitarian cost dominance ratio
    print("\n\n  5. HUMANITARIAN VS ECONOMIC COST DOMINANCE")
    print("  " + "-" * 60)
    for region, df in all_sim_dfs.items():
        opt_idx = df["total_costs"].idxmin()
        opt = df.loc[opt_idx]
        ratio = opt["humanitarian_costs"] / max(opt["economic_costs"], 1)
        print(f"    {region}: humanitarian/economic ratio = {ratio:.1f}x "
              f"(humanitarian ${opt['humanitarian_costs']/1e9:.1f}B, "
              f"economic ${opt['economic_costs']/1e9:.1f}B)")

    insights["marginal_costs"] = marginal_data
    insights["month2_sensitivity"] = sensitivity_data
    insights["lockdown_crossover"] = crossover_data

    with open(output_dir / "novel_insights.json", "w") as f:
        json.dump(insights, f, indent=2, default=str)

    # Plot: marginal severity cost curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    regions = list(all_sim_dfs.keys())[:4]
    for idx, region in enumerate(regions):
        ax = axes.flat[idx]
        rdf = mdf[mdf["region"] == region].sort_values("severity")
        gdp = TOTAL_GDP.get(region, 1e12)
        sevs = rdf["severity"].values
        econ_pct = (rdf["avg_econ"] / gdp * 100).values
        human_pct = (rdf["avg_human"] / gdp * 100).values
        total_pct = ((rdf["avg_econ"] + rdf["avg_human"]) / gdp * 100).values

        ax.plot(sevs, econ_pct, "s-", color="#4e79a7", label="Economic",
                markersize=7)
        ax.plot(sevs, human_pct, "^-", color="#e15759", label="Humanitarian",
                markersize=7)
        ax.plot(sevs, total_pct, "o-", color="#333333", label="Total",
                markersize=7, linewidth=2)

        npi_labels = rdf["npi"].values
        for i, (s, t) in enumerate(zip(sevs, total_pct)):
            ax.annotate(npi_labels[i], (s, t), textcoords="offset points",
                        xytext=(0, 10), fontsize=7, ha="center", color="gray")

        if len(total_pct) >= 2:
            min_idx = np.argmin(total_pct)
            ax.axvline(sevs[min_idx], color="green", alpha=0.5, linestyle="--",
                       linewidth=1)
            ax.text(sevs[min_idx] + 0.1, max(total_pct) * 0.9,
                    f"Optimal\nsev={sevs[min_idx]:.0f}",
                    fontsize=8, color="green")

        ax.set_xlabel("Month-1 NPI Severity", fontsize=10)
        ax.set_ylabel("Avg Cost (% of GDP)", fontsize=10)
        ax.set_title(f"{region}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xticks(range(6))

    plt.tight_layout(pad=2.0)
    fig.savefig(output_dir / "marginal_severity_costs.pdf",
                bbox_inches="tight", dpi=200)
    fig.savefig(output_dir / "marginal_severity_costs.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)

    return insights


SEVERITY_TO_POLICY = {v: k for k, v in POLICY_SEVERITY.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Extract decision rules from THEMIS simulations")
    parser.add_argument("--regions", nargs="+", default=REGIONS)
    parser.add_argument("--startdate", default="2020-03-15")
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--output-dir",
                        default="simulation_results/decision_rules")
    args = parser.parse_args()

    run_decision_rule_extraction(
        regions=args.regions,
        start_date=args.startdate,
        policy_length=args.length,
        max_depth=args.depth,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
