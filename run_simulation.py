from pathlib import Path

import pandas as pd

from config import (
    OUTPUT_DIR,
    OUTPUT_BASE_FILE,
    OUTPUT_SUMMARY_FILE,
    OUTPUT_ABRUPT_FILE,
    OUTPUT_GRADUAL_FILE,
    OUTPUT_VOLATILITY_FILE,
    INITIAL_INVENTORY_MULTIPLIER,
    TARGET_STOCK_MULTIPLIER,
    REORDER_QUANTILE,
    ROLLING_WINDOW,
    NOMINAL_COVERAGE,
    ABRUPT_SURGE_FACTORS,
    ABRUPT_SURGE_MONTHS,
    GRADUAL_DRIFT_TOTAL_INCREASES,
    GRADUAL_DRIFT_MONTHS,
    VOLATILITY_MULTIPLIERS,
    LEAD_TIME_DELAY_MONTHS,
    RANDOM_SEED,
)

from data_loader import (
    load_raw_data,
    filter_series,
    create_continuous_monthly_series,
)

from feature_engineering import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    add_intermittent_flag,
    assign_clinical_priority,
)

from demand_scenarios import (
    create_stable_scenario,
    create_abrupt_surge_scenario,
    create_gradual_drift_scenario,
    create_volatility_scenario,
)

from inventory_simulation import (
    assign_lead_times,
    calculate_initial_inventory,
    simulate_inventory,
)

from reliability_monitoring import add_reliability_columns


def ensure_output_directory():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_base_data():
    print("Loading NHSBSA dataset...")

    raw_df = load_raw_data()
    print(f"Raw rows loaded: {len(raw_df):,}")

    filtered_df = filter_series(raw_df)
    print(f"Rows after history filtering: {len(filtered_df):,}")

    monthly_df = create_continuous_monthly_series(filtered_df)
    print(f"Rows after monthly alignment: {len(monthly_df):,}")

    monthly_df = add_time_features(monthly_df)
    monthly_df = add_lag_features(monthly_df)
    monthly_df = add_rolling_features(monthly_df)
    monthly_df = add_intermittent_flag(monthly_df)
    monthly_df = assign_clinical_priority(monthly_df)

    monthly_df = create_stable_scenario(monthly_df)

    monthly_df = assign_lead_times(
        monthly_df,
        seed=RANDOM_SEED,
    )

    monthly_df = calculate_initial_inventory(
        monthly_df,
        multiplier=INITIAL_INVENTORY_MULTIPLIER,
    )

    monthly_df = simulate_inventory(
        monthly_df,
        target_stock_multiplier=TARGET_STOCK_MULTIPLIER,
        reorder_quantile=REORDER_QUANTILE,
    )

    monthly_df = add_reliability_columns(
        monthly_df,
        window=ROLLING_WINDOW,
        nominal_coverage=NOMINAL_COVERAGE,
    )

    return monthly_df


def save_base_data(df):
    df.to_csv(OUTPUT_BASE_FILE, index=False)
    print(f"Saved base simulation: {OUTPUT_BASE_FILE}")


def create_and_save_scenarios(base_df):
    scenario_outputs = []

    print("Creating abrupt surge scenarios...")

    for factor in ABRUPT_SURGE_FACTORS:
        scenario_df = create_abrupt_surge_scenario(
            base_df,
            surge_factor=factor,
            surge_months=ABRUPT_SURGE_MONTHS,
        )

        scenario_df = simulate_inventory(
            scenario_df,
            target_stock_multiplier=TARGET_STOCK_MULTIPLIER,
            reorder_quantile=REORDER_QUANTILE,
        )

        scenario_df = add_reliability_columns(
            scenario_df,
            window=ROLLING_WINDOW,
            nominal_coverage=NOMINAL_COVERAGE,
        )

        output_file = OUTPUT_DIR / (
            f"scenario_abrupt_surge_{str(factor).replace('.', '_')}.csv"
        )

        scenario_df.to_csv(output_file, index=False)

        scenario_outputs.append(
            {
                "scenario": f"ABRUPT_SURGE_{factor}",
                "rows": len(scenario_df),
                "stockout_rate": scenario_df["STOCKOUT_EVENT"].mean(),
                "output_file": str(output_file),
            }
        )

    print("Creating gradual drift scenarios...")

    for increase in GRADUAL_DRIFT_TOTAL_INCREASES:
        scenario_df = create_gradual_drift_scenario(
            base_df,
            total_increase=increase,
            drift_months=GRADUAL_DRIFT_MONTHS,
        )

        scenario_df = simulate_inventory(
            scenario_df,
            target_stock_multiplier=TARGET_STOCK_MULTIPLIER,
            reorder_quantile=REORDER_QUANTILE,
        )

        scenario_df = add_reliability_columns(
            scenario_df,
            window=ROLLING_WINDOW,
            nominal_coverage=NOMINAL_COVERAGE,
        )

        output_file = OUTPUT_DIR / (
            f"scenario_gradual_drift_{str(increase).replace('.', '_')}.csv"
        )

        scenario_df.to_csv(output_file, index=False)

        scenario_outputs.append(
            {
                "scenario": f"GRADUAL_DRIFT_{increase}",
                "rows": len(scenario_df),
                "stockout_rate": scenario_df["STOCKOUT_EVENT"].mean(),
                "output_file": str(output_file),
            }
        )

    print("Creating volatility scenarios...")

    for multiplier in VOLATILITY_MULTIPLIERS:
        scenario_df = create_volatility_scenario(
            base_df,
            volatility_multiplier=multiplier,
        )

        scenario_df = simulate_inventory(
            scenario_df,
            target_stock_multiplier=TARGET_STOCK_MULTIPLIER,
            reorder_quantile=REORDER_QUANTILE,
        )

        scenario_df = add_reliability_columns(
            scenario_df,
            window=ROLLING_WINDOW,
            nominal_coverage=NOMINAL_COVERAGE,
        )

        output_file = OUTPUT_DIR / (
            f"scenario_volatility_{str(multiplier).replace('.', '_')}.csv"
        )

        scenario_df.to_csv(output_file, index=False)

        scenario_outputs.append(
            {
                "scenario": f"VOLATILITY_{multiplier}",
                "rows": len(scenario_df),
                "stockout_rate": scenario_df["STOCKOUT_EVENT"].mean(),
                "output_file": str(output_file),
            }
        )

    summary_df = pd.DataFrame(scenario_outputs)
    summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)

    print(f"Saved scenario summary: {OUTPUT_SUMMARY_FILE}")


def main():
    ensure_output_directory()

    base_df = prepare_base_data()

    save_base_data(base_df)

    create_and_save_scenarios(base_df)

    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()