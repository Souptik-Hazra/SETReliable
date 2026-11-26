import pandas as pd

from config import (
    OUTPUT_DIR, GROSS_TARGET, NET_TARGET, FORECAST_TARGETS,
    MIN_HISTORY_MONTHS, TARGET_STOCK_MULTIPLIER, REORDER_QUANTILE,
    ROLLING_WINDOW, NOMINAL_COVERAGE, ABRUPT_SURGE_FACTORS,
    ABRUPT_SURGE_MONTHS, GRADUAL_DRIFT_TOTAL_INCREASES,
    GRADUAL_DRIFT_MONTHS, VOLATILITY_MULTIPLIERS, RANDOM_SEED,
    APPLY_NEGATIVE_MOVEMENTS_AS_RETURNS,
)
from data_loader import load_raw_data, filter_series, create_continuous_monthly_series
from feature_engineering import add_time_features, add_target_features, add_intermittent_flag
from demand_scenarios import (
    create_stable_scenario, create_abrupt_surge_scenario,
    create_gradual_drift_scenario, create_volatility_scenario,
)
from inventory_simulation import (
    assign_lead_times, calculate_initial_inventory, simulate_inventory,
)
from reliability_monitoring import add_reliability_columns, evaluate_forecast_metrics


def validate_forecast_frame(frame, target):
    required = {target, "BASELINE_FORECAST", "FORECAST_RESIDUAL", "RELIABILITY_SCORE"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Forecast output missing columns: {sorted(missing)}")
    if frame["RELIABILITY_SCORE"].isna().any():
        raise ValueError("Reliability score unexpectedly contains missing values")
    return frame


def run_target_experiment(monthly, target):
    featured = add_target_features(monthly, target)
    featured = add_intermittent_flag(featured, target)
    evaluated = add_reliability_columns(
        featured, target=target, window=ROLLING_WINDOW,
        nominal_coverage=NOMINAL_COVERAGE,
    )
    validate_forecast_frame(evaluated, target)
    metrics = evaluate_forecast_metrics(evaluated, target)
    evaluated.to_csv(OUTPUT_DIR / f"forecast_{target.lower()}.csv", index=False)
    return metrics


def run_gross_inventory_scenarios(monthly):
    inventory_input = assign_lead_times(monthly, seed=RANDOM_SEED)
    inventory_input = calculate_initial_inventory(inventory_input, initial_inventory=0.0)
    stable = create_stable_scenario(inventory_input, target=GROSS_TARGET)
    scenarios = [("stable_gross", stable)]
    for factor in ABRUPT_SURGE_FACTORS:
        scenarios.append((f"surge_{factor}", create_abrupt_surge_scenario(
            inventory_input, target=GROSS_TARGET, surge_factor=factor,
            surge_months=ABRUPT_SURGE_MONTHS)))
    for increase in GRADUAL_DRIFT_TOTAL_INCREASES:
        scenarios.append((f"drift_{increase}", create_gradual_drift_scenario(
            inventory_input, target=GROSS_TARGET, total_increase=increase,
            drift_months=GRADUAL_DRIFT_MONTHS)))
    for multiplier in VOLATILITY_MULTIPLIERS:
        scenarios.append((f"volatility_{multiplier}", create_volatility_scenario(
            inventory_input, target=GROSS_TARGET,
            volatility_multiplier=multiplier)))

    summaries = []
    for name, scenario in scenarios:
        result = simulate_inventory(
            scenario, target_stock_multiplier=TARGET_STOCK_MULTIPLIER,
            reorder_quantile=REORDER_QUANTILE,
            apply_negative_movements_as_returns=APPLY_NEGATIVE_MOVEMENTS_AS_RETURNS,
        )
        result.to_csv(OUTPUT_DIR / f"inventory_{name}.csv", index=False)
        observed = result["SCENARIO_QUANTITY"].notna()
        summaries.append({
            "scenario": name,
            "target": GROSS_TARGET,
            "rows": len(result),
            "observed_months": int(observed.sum()),
            "stockout_rate": float(result.loc[observed, "STOCKOUT_EVENT"].mean())
                if observed.any() else float("nan"),
            "total_shortage_native_units": float(result["SHORTAGE_QUANTITY"].sum()),
            "negative_movements_treated_as_returns": APPLY_NEGATIVE_MOVEMENTS_AS_RETURNS,
            "results_are_hypothetical": True,
        })
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "inventory_scenario_summary.csv", index=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and aggregating the current NHSBSA SCMD dataset...")
    raw = load_raw_data()
    quality = raw.attrs.get("quality_report", {})
    pd.DataFrame([quality]).to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)
    print(f"Aggregated rows: {len(raw):,}")
    print(f"Data-quality summary: {quality}")

    # Eligibility is based on gross-positive observed months, not net values.
    eligible = filter_series(raw, min_months=MIN_HISTORY_MONTHS, target=GROSS_TARGET)
    monthly = create_continuous_monthly_series(eligible)
    monthly = add_time_features(monthly)
    monthly.to_csv(OUTPUT_DIR / "monthly_observed_measures.csv", index=False)

    metrics = [run_target_experiment(monthly, target) for target in FORECAST_TARGETS]
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / "forecast_comparison.csv", index=False)
    run_gross_inventory_scenarios(monthly)
    print("Completed. Forecast metrics compare separate gross and net targets.")
    print("Inventory and stress-test results are hypothetical, assumption-driven scenarios.")


if __name__ == "__main__":
    main()
