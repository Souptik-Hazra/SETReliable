import pandas as pd

from config import (
    OUTPUT_DIR, OUTPUT_BASE_FILE, OUTPUT_SUMMARY_FILE,
    INITIAL_INVENTORY_MULTIPLIER, TARGET_STOCK_MULTIPLIER,
    REORDER_QUANTILE, ROLLING_WINDOW, NOMINAL_COVERAGE,
    ABRUPT_SURGE_FACTORS, ABRUPT_SURGE_MONTHS,
    GRADUAL_DRIFT_TOTAL_INCREASES, GRADUAL_DRIFT_MONTHS,
    VOLATILITY_MULTIPLIERS, RANDOM_SEED,
)
from data_loader import load_raw_data, filter_series, create_continuous_monthly_series
from feature_engineering import (
    add_time_features, add_lag_features, add_rolling_features,
    add_intermittent_flag, assign_clinical_priority,
)
from demand_scenarios import (
    create_stable_scenario, create_abrupt_surge_scenario,
    create_gradual_drift_scenario, create_volatility_scenario,
)
from inventory_simulation import (
    assign_lead_times, calculate_initial_inventory, simulate_inventory,
)
from reliability_monitoring import add_reliability_columns


def prepare_base_data():
    print("Loading NHSBSA secondary-care medicines dataset...")
    raw = load_raw_data()
    print(f"Monthly medicine-Trust-unit observations: {len(raw):,}")

    eligible = filter_series(raw)
    print(f"Eligible series observations: {len(eligible):,}")

    monthly = create_continuous_monthly_series(eligible)
    monthly = add_time_features(monthly)
    monthly = add_lag_features(monthly)
    monthly = add_rolling_features(monthly)
    monthly = add_intermittent_flag(monthly)
    monthly = assign_clinical_priority(monthly)
    monthly = create_stable_scenario(monthly)
    monthly = assign_lead_times(monthly, seed=RANDOM_SEED)
    monthly = calculate_initial_inventory(
        monthly, multiplier=INITIAL_INVENTORY_MULTIPLIER
    )
    monthly = simulate_inventory(
        monthly,
        target_stock_multiplier=TARGET_STOCK_MULTIPLIER,
        reorder_quantile=REORDER_QUANTILE,
    )
    monthly = add_reliability_columns(
        monthly, window=ROLLING_WINDOW, nominal_coverage=NOMINAL_COVERAGE
    )
    return monthly


def _run_scenario(base_df, scenario_name, scenario_df):
    result = simulate_inventory(
        scenario_df,
        target_stock_multiplier=TARGET_STOCK_MULTIPLIER,
        reorder_quantile=REORDER_QUANTILE,
    )
    result = add_reliability_columns(
        result, window=ROLLING_WINDOW, nominal_coverage=NOMINAL_COVERAGE
    )
    output = OUTPUT_DIR / f"{scenario_name.lower()}.csv"
    result.to_csv(output, index=False)

    observed = result["DEMAND_SCENARIO"].notna()
    summary = {
        "scenario": scenario_name,
        "rows": len(result),
        "observed_months": int(observed.sum()),
        "stockout_rate": float(result.loc[observed, "STOCKOUT_EVENT"].mean())
            if observed.any() else float("nan"),
        "total_shortage_quantity_native_units": float(result["SHORTAGE_QUANTITY"].sum()),
        "output_file": str(output),
    }
    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = prepare_base_data()
    base.to_csv(OUTPUT_BASE_FILE, index=False)
    print(f"Saved baseline: {OUTPUT_BASE_FILE}")

    summaries = []
    for factor in ABRUPT_SURGE_FACTORS:
        scenario = create_abrupt_surge_scenario(
            base, surge_factor=factor, surge_months=ABRUPT_SURGE_MONTHS
        )
        summaries.append(_run_scenario(base, f"ABRUPT_SURGE_{factor}", scenario))

    for increase in GRADUAL_DRIFT_TOTAL_INCREASES:
        scenario = create_gradual_drift_scenario(
            base, total_increase=increase, drift_months=GRADUAL_DRIFT_MONTHS
        )
        summaries.append(_run_scenario(base, f"GRADUAL_DRIFT_{increase}", scenario))

    for multiplier in VOLATILITY_MULTIPLIERS:
        scenario = create_volatility_scenario(
            base, volatility_multiplier=multiplier
        )
        summaries.append(_run_scenario(base, f"VOLATILITY_{multiplier}", scenario))

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_SUMMARY_FILE, index=False)
    print(f"Saved scenario summary: {OUTPUT_SUMMARY_FILE}")
    print("Simulation completed.")


if __name__ == "__main__":
    main()
