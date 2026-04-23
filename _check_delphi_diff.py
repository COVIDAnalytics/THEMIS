"""Compare DELPHI vs DELPHI-constant daily predictions to find the bug."""
import pandas as pd

run = "simulation_results/themis_time_transfer_20260422_220348"
daily = pd.read_csv(f"{run}/daily_predictions_ape_detail.csv")

for region in daily["region"].unique():
    rd = daily[daily["region"] == region]
    for model in ["delphi", "delphi_constant", "themis"]:
        m = rd[rd["model"] == model]
        if m.empty:
            continue
        print(f"{region:20s} | {model:18s} | cases_start={m['pred_cases'].iloc[0]:12.1f} cases_end={m['pred_cases'].iloc[-1]:12.1f} | deaths_end={m['pred_deaths'].iloc[-1]:12.1f}")
    print()

# Also check the raw simulations for one region
print("\n=== Detailed day-by-day for Germany ===")
rg = daily[(daily["region"] == "Germany")]
for model in ["delphi", "delphi_constant"]:
    m = rg[rg["model"] == model].sort_values("date")
    print(f"\n{model}:")
    print(m[["date", "pred_cases", "actual_cases", "pred_deaths"]].head(5).to_string(index=False))
    print("...")
    print(m[["date", "pred_cases", "actual_cases", "pred_deaths"]].tail(5).to_string(index=False))
