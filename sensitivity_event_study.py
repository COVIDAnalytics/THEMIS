"""
Event-study style falsification test for THEMIS gamma estimates.

Tests whether NPI policy transitions cause *discontinuities* in the fitted
gamma(t) curve, which they should if gamma captures the causal NPI effect.

Two complementary approaches:

1. EVENT-STUDY REGRESSION: For each policy transition event, estimate
       gamma(t) = alpha + beta_post * Post_t + sum_k beta_k * (t - t0)^k + eps
   where Post_t = 1 if t >= t0 (transition date).  A significant beta_post
   indicates a discrete jump at the policy transition -- evidence that
   gamma responds to NPI timing, not just smooth trends.

2. GRANGER CAUSALITY: Test whether lagged policy indicators Granger-cause
   gamma(t), controlling for lagged gamma.  Significant F-tests support
   the causal direction NPI -> gamma.

3. PLACEBO TRANSITIONS: Randomly reassign transition dates and re-run the
   event-study regression.  The true transition should produce a larger
   |beta_post| than placebo transitions.

Usage:
    python sensitivity_event_study.py
"""
import argparse
import json
import warnings
from datetime import timedelta
from pathlib import Path

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


def _active_policy_label(row, policy_cols):
    for p in policy_cols:
        if row.get(p, 0) == 1:
            return p
    return None


def build_gamma_panel(start_date="2020-03-01", end_date="2020-07-31"):
    """Build a panel: (region, date, gamma, active_policy, severity_rank)."""
    params_unique = (
        past_parameters
        .sort_values(["Country", "Province", "Data Start Date"])
        .drop_duplicates(subset=["Country", "Province"], keep="last")
        .reset_index(drop=True)
    )

    severity_order = {
        "No_Measure": 1,
        "Restrict_Mass_Gatherings": 2,
        "Mass_Gatherings_Authorized_But_Others_Restricted": 3,
        "Restrict_Mass_Gatherings_and_Schools": 4,
        "Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others": 5,
        "Restrict_Mass_Gatherings_and_Schools_and_Others": 6,
        "Lockdown": 7,
    }

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

        policy_cols = [p for p in POLICY_NAMES if p in policy_data.columns]
        if not policy_cols:
            continue

        for _, drow in policy_data.iterrows():
            g = float(gamma_t(drow["date"], params_list))
            pol = _active_policy_label(drow, policy_cols)
            rows.append({
                "region": region_id,
                "date": drow["date"],
                "gamma": g,
                "policy": pol,
                "severity": severity_order.get(pol, 0),
            })

    return pd.DataFrame(rows)


def find_transition_events(panel, min_pre_days=7, min_post_days=7):
    """Identify policy transition events: dates where the active policy changes."""
    events = []
    for region, grp in panel.groupby("region"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for i in range(1, len(grp)):
            if grp.loc[i, "policy"] != grp.loc[i - 1, "policy"]:
                t0 = grp.loc[i, "date"]
                pre = grp[(grp["date"] < t0) & (grp["date"] >= t0 - timedelta(days=30))]
                post = grp[(grp["date"] >= t0) & (grp["date"] < t0 + timedelta(days=30))]
                if len(pre) >= min_pre_days and len(post) >= min_post_days:
                    severity_change = grp.loc[i, "severity"] - grp.loc[i - 1, "severity"]
                    events.append({
                        "region": region,
                        "t0": t0,
                        "policy_before": grp.loc[i - 1, "policy"],
                        "policy_after": grp.loc[i, "policy"],
                        "severity_change": severity_change,
                        "direction": "tightening" if severity_change > 0 else "relaxing",
                    })
    return pd.DataFrame(events)


def event_study_regression(panel, events, window=21):
    """
    For each transition event, estimate:
        gamma(t) = alpha + beta_post * Post + beta_trend * (t - t0) + eps
    in a symmetric window around t0.
    Returns beta_post estimates and their significance.
    """
    results = []
    for _, ev in events.iterrows():
        region = ev["region"]
        t0 = ev["t0"]
        grp = panel[panel["region"] == region].copy()
        grp = grp[(grp["date"] >= t0 - timedelta(days=window)) &
                  (grp["date"] < t0 + timedelta(days=window))]

        if len(grp) < 10:
            continue

        grp = grp.sort_values("date").reset_index(drop=True)
        grp["t"] = (grp["date"] - t0).dt.days
        grp["post"] = (grp["t"] >= 0).astype(int)

        X = np.column_stack([
            np.ones(len(grp)),
            grp["post"].values,
            grp["t"].values,
        ])
        y = grp["gamma"].values

        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            if len(residuals) == 0:
                sse = np.sum((y - X @ beta) ** 2)
            else:
                sse = residuals[0]
            n = len(y)
            k = X.shape[1]
            mse = sse / max(n - k, 1)
            XtX_inv = np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(np.diag(XtX_inv) * mse)
            t_stat = beta[1] / se_beta[1] if se_beta[1] > 1e-12 else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), max(n - k, 1)))

            results.append({
                "region": region,
                "t0": str(t0.date()),
                "policy_before": ev["policy_before"],
                "policy_after": ev["policy_after"],
                "direction": ev["direction"],
                "severity_change": ev["severity_change"],
                "beta_post": float(beta[1]),
                "se_beta_post": float(se_beta[1]),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "n_obs": n,
                "r_squared": float(1 - sse / np.sum((y - y.mean()) ** 2)) if np.sum((y - y.mean()) ** 2) > 0 else 0,
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def placebo_test(panel, events, window=21, n_placebo=200):
    """
    For each real transition, randomly re-assign the transition date within
    the same region's timeline and re-run the event-study regression.
    Compare |beta_post| from real vs placebo transitions.
    """
    real_results = event_study_regression(panel, events, window)
    if len(real_results) == 0:
        return real_results, pd.DataFrame()

    placebo_betas = []
    for _ in range(n_placebo):
        fake_events = events.copy()
        for idx in fake_events.index:
            region = fake_events.loc[idx, "region"]
            grp = panel[panel["region"] == region]
            valid_dates = grp["date"].values
            if len(valid_dates) > 2 * window:
                rand_idx = np.random.randint(window, len(valid_dates) - window)
                fake_events.loc[idx, "t0"] = pd.Timestamp(valid_dates[rand_idx])

        fake_results = event_study_regression(panel, fake_events, window)
        if len(fake_results) > 0:
            placebo_betas.append(fake_results["beta_post"].abs().mean())

    placebo_df = pd.DataFrame({"abs_beta_post_mean": placebo_betas})
    return real_results, placebo_df


def granger_causality_test(panel, max_lag=7):
    """
    For each region, test whether the policy severity indicator Granger-causes
    gamma(t).  Compare restricted (gamma ~ own lags) vs unrestricted
    (gamma ~ own lags + policy lags) models.
    """
    results = []
    for region, grp in panel.groupby("region"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < max_lag * 3:
            continue

        g = grp["gamma"].values
        s = grp["severity"].values.astype(float)

        y = g[max_lag:]
        X_restricted = np.column_stack([g[max_lag - k:-k] for k in range(1, max_lag + 1)])
        X_unrestricted = np.column_stack([
            *[g[max_lag - k:-k] for k in range(1, max_lag + 1)],
            *[s[max_lag - k:-k] for k in range(1, max_lag + 1)],
        ])

        X_r = np.column_stack([np.ones(len(y)), X_restricted])
        X_u = np.column_stack([np.ones(len(y)), X_unrestricted])

        try:
            beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
            sse_r = np.sum((y - X_r @ beta_r) ** 2)
            sse_u = np.sum((y - X_u @ beta_u) ** 2)
            n = len(y)
            q = max_lag  # number of restrictions
            k_u = X_u.shape[1]
            f_stat = ((sse_r - sse_u) / q) / (sse_u / max(n - k_u, 1))
            p_val = 1 - stats.f.cdf(f_stat, q, max(n - k_u, 1))

            results.append({
                "region": region,
                "n_obs": n,
                "f_stat": float(f_stat),
                "p_value": float(p_val),
                "sse_restricted": float(sse_r),
                "sse_unrestricted": float(sse_u),
                "significant_5pct": p_val < 0.05,
                "significant_1pct": p_val < 0.01,
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def compute_directional_test(results):
    """
    Test whether tightening transitions produce negative beta_post
    and relaxing transitions produce positive beta_post (sign test).
    """
    tight = results[results["direction"] == "tightening"]["beta_post"]
    relax = results[results["direction"] == "relaxing"]["beta_post"]

    out = {}
    if len(tight) > 0:
        frac_neg = (tight < 0).mean()
        binom_p = stats.binom_test(int((tight < 0).sum()), len(tight), 0.5,
                                   alternative="greater") if hasattr(stats, "binom_test") else \
                  stats.binomtest(int((tight < 0).sum()), len(tight), 0.5,
                                 alternative="greater").pvalue
        out["tightening_n"] = len(tight)
        out["tightening_frac_negative"] = float(frac_neg)
        out["tightening_sign_test_p"] = float(binom_p)
        ttest = stats.ttest_1samp(tight, 0, alternative="less")
        out["tightening_ttest_stat"] = float(ttest.statistic)
        out["tightening_ttest_p"] = float(ttest.pvalue)

    if len(relax) > 0:
        frac_pos = (relax > 0).mean()
        binom_p = stats.binom_test(int((relax > 0).sum()), len(relax), 0.5,
                                   alternative="greater") if hasattr(stats, "binom_test") else \
                  stats.binomtest(int((relax > 0).sum()), len(relax), 0.5,
                                 alternative="greater").pvalue
        out["relaxing_n"] = len(relax)
        out["relaxing_frac_positive"] = float(frac_pos)
        out["relaxing_sign_test_p"] = float(binom_p)
        ttest = stats.ttest_1samp(relax, 0, alternative="greater")
        out["relaxing_ttest_stat"] = float(ttest.statistic)
        out["relaxing_ttest_p"] = float(ttest.pvalue)

    return out


def plot_event_study(results, output_dir):
    """Plot distribution of beta_post by direction (tightening vs relaxing)."""
    if len(results) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    tightening = results[results["direction"] == "tightening"]["beta_post"]
    relaxing = results[results["direction"] == "relaxing"]["beta_post"]
    if len(tightening) > 0:
        ax.hist(tightening, bins=25, alpha=0.7, color="#d95f02", label=f"Tightening (n={len(tightening)})", edgecolor="white")
    if len(relaxing) > 0:
        ax.hist(relaxing, bins=25, alpha=0.7, color="#1b9e77", label=f"Relaxing (n={len(relaxing)})", edgecolor="white")
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("$\\hat{\\beta}_{post}$ (jump in $\\gamma$ at transition)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Event-Study: $\\gamma$ Discontinuity at Policy Transitions", fontsize=12)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.scatter(results["severity_change"], results["beta_post"],
               s=15, alpha=0.5, c="#2c7fb8")
    slope, intercept, r_val, p_val, se = stats.linregress(
        results["severity_change"], results["beta_post"])
    x_line = np.linspace(results["severity_change"].min(), results["severity_change"].max(), 50)
    ax.plot(x_line, intercept + slope * x_line, "r-", alpha=0.7,
            label=f"slope={slope:.4f}, r={r_val:.3f}, p={p_val:.2e}")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Severity Change (+ = tightening)", fontsize=11)
    ax.set_ylabel("$\\hat{\\beta}_{post}$", fontsize=11)
    ax.set_title("Dose-Response: Severity Change vs $\\gamma$ Jump", fontsize=12)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(results["p_value"], bins=20, color="#7570b3", edgecolor="white", alpha=0.8)
    ax.axvline(x=0.05, color="red", linestyle="--", alpha=0.7, label="p = 0.05")
    sig_pct = (results["p_value"] < 0.05).mean() * 100
    ax.set_xlabel("p-value", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Significance of $\\beta_{{post}}$ ({sig_pct:.0f}% significant at 5%)", fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "event_study_results.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "event_study_results.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_placebo(real_results, placebo_df, output_dir):
    """Compare real |beta_post| to placebo distribution."""
    if len(real_results) == 0 or len(placebo_df) == 0:
        return

    real_mean = real_results["beta_post"].abs().mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(placebo_df["abs_beta_post_mean"], bins=30, color="#a6cee3",
            edgecolor="white", alpha=0.8, label="Placebo transitions")
    ax.axvline(x=real_mean, color="red", linewidth=2, linestyle="-",
               label=f"Real transitions ({real_mean:.4f})")
    pval = (placebo_df["abs_beta_post_mean"] >= real_mean).mean()
    ax.set_xlabel("Mean $|\\hat{\\beta}_{post}|$", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Placebo Test: Real vs. Random Transitions (p = {pval:.3f})", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "placebo_test.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "placebo_test.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Event-study falsification tests")
    parser.add_argument("--start-date", default="2020-03-01")
    parser.add_argument("--end-date", default="2020-07-31")
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--n-placebo", type=int, default=200)
    parser.add_argument("--granger-lags", type=int, default=7)
    parser.add_argument("--output-dir", default="simulation_results/event_study")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  EVENT-STUDY FALSIFICATION TESTS")
    print("=" * 70)

    # Build panel
    print("\n[1] Building gamma panel across all regions ...")
    panel = build_gamma_panel(args.start_date, args.end_date)
    n_regions = panel["region"].nunique()
    print(f"  Panel: {len(panel)} obs, {n_regions} regions")
    panel.to_csv(output_dir / "gamma_panel.csv", index=False)

    # Find transition events
    print("\n[2] Identifying policy transition events ...")
    events = find_transition_events(panel)
    print(f"  Found {len(events)} transition events")
    if len(events) > 0:
        print(f"  Tightening: {(events['direction'] == 'tightening').sum()}")
        print(f"  Relaxing: {(events['direction'] == 'relaxing').sum()}")
        events.to_csv(output_dir / "transition_events.csv", index=False)

    # Event-study regression
    print("\n[3] Event-study regressions (window = +/- {0} days) ...".format(args.window))
    es_results = event_study_regression(panel, events, window=args.window)
    if len(es_results) > 0:
        sig_5 = (es_results["p_value"] < 0.05).mean() * 100
        sig_1 = (es_results["p_value"] < 0.01).mean() * 100
        print(f"  {len(es_results)} regressions estimated")
        print(f"  Significant at 5%: {sig_5:.1f}%")
        print(f"  Significant at 1%: {sig_1:.1f}%")
        print(f"  Mean |beta_post|: {es_results['beta_post'].abs().mean():.4f}")

        tightening = es_results[es_results["direction"] == "tightening"]
        relaxing = es_results[es_results["direction"] == "relaxing"]
        if len(tightening) > 0:
            print(f"  Tightening: mean beta_post = {tightening['beta_post'].mean():.4f} "
                  f"({(tightening['p_value'] < 0.05).mean()*100:.0f}% sig)")
        if len(relaxing) > 0:
            print(f"  Relaxing:   mean beta_post = {relaxing['beta_post'].mean():.4f} "
                  f"({(relaxing['p_value'] < 0.05).mean()*100:.0f}% sig)")

        slope, intercept, r_val, p_val_slope, se = stats.linregress(
            es_results["severity_change"], es_results["beta_post"])
        print(f"\n  Dose-response (beta_post ~ severity_change):")
        print(f"    slope = {slope:.4f}, r = {r_val:.3f}, p = {p_val_slope:.2e}")
        if slope < 0 and p_val_slope < 0.05:
            print(f"    >>> SIGNIFICANT negative slope: tightening reduces gamma, relaxing increases it")

        directional = compute_directional_test(es_results)
        print(f"\n  Directional tests:")
        if "tightening_frac_negative" in directional:
            print(f"    Tightening: {directional['tightening_frac_negative']*100:.1f}% have beta_post < 0 "
                  f"(sign-test p = {directional['tightening_sign_test_p']:.2e}, "
                  f"t-test p = {directional['tightening_ttest_p']:.2e})")
        if "relaxing_frac_positive" in directional:
            print(f"    Relaxing:   {directional['relaxing_frac_positive']*100:.1f}% have beta_post > 0 "
                  f"(sign-test p = {directional['relaxing_sign_test_p']:.2e}, "
                  f"t-test p = {directional['relaxing_ttest_p']:.2e})")

        es_results.to_csv(output_dir / "event_study_regressions.csv", index=False)
        plot_event_study(es_results, output_dir)

    # Placebo test
    print(f"\n[4] Placebo test ({args.n_placebo} random reassignments) ...")
    real_results, placebo_df = placebo_test(panel, events, window=args.window, n_placebo=args.n_placebo)
    if len(real_results) > 0 and len(placebo_df) > 0:
        real_mean = real_results["beta_post"].abs().mean()
        placebo_p = (placebo_df["abs_beta_post_mean"] >= real_mean).mean()
        print(f"  Real mean |beta_post| = {real_mean:.4f}")
        print(f"  Placebo p-value = {placebo_p:.3f}")
        if placebo_p < 0.05:
            print(f"  >>> SIGNIFICANT: real transitions produce larger gamma jumps than random dates")
        placebo_df.to_csv(output_dir / "placebo_betas.csv", index=False)
        plot_placebo(real_results, placebo_df, output_dir)

    # Granger causality
    print(f"\n[5] Granger causality tests (max lag = {args.granger_lags}) ...")
    granger_results = granger_causality_test(panel, max_lag=args.granger_lags)
    if len(granger_results) > 0:
        sig_5 = (granger_results["significant_5pct"]).mean() * 100
        sig_1 = (granger_results["significant_1pct"]).mean() * 100
        print(f"  {len(granger_results)} regions tested")
        print(f"  Significant at 5%: {sig_5:.1f}% of regions")
        print(f"  Significant at 1%: {sig_1:.1f}% of regions")
        print(f"  Median F-stat: {granger_results['f_stat'].median():.2f}")
        granger_results.to_csv(output_dir / "granger_causality.csv", index=False)

    # Summary
    summary = {
        "panel_size": len(panel),
        "n_regions": int(n_regions),
        "n_transitions": len(events),
    }
    if len(es_results) > 0:
        summary["event_study"] = {
            "n_regressions": len(es_results),
            "pct_significant_5pct": float((es_results["p_value"] < 0.05).mean() * 100),
            "pct_significant_1pct": float((es_results["p_value"] < 0.01).mean() * 100),
            "mean_abs_beta_post": float(es_results["beta_post"].abs().mean()),
            "dose_response_slope": float(slope),
            "dose_response_r": float(r_val),
            "dose_response_p": float(p_val_slope),
            "directional_tests": directional if 'directional' in dir() else {},
        }
    if len(real_results) > 0 and len(placebo_df) > 0:
        summary["placebo_test"] = {
            "real_mean_abs_beta": float(real_mean),
            "placebo_p_value": float(placebo_p),
            "n_placebo": args.n_placebo,
        }
    if len(granger_results) > 0:
        summary["granger_causality"] = {
            "n_regions_tested": len(granger_results),
            "pct_significant_5pct": float((granger_results["significant_5pct"]).mean() * 100),
            "pct_significant_1pct": float((granger_results["significant_1pct"]).mean() * 100),
            "median_f_stat": float(granger_results["f_stat"].median()),
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  All outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
