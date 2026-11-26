"""Build unit-aware medicine-Trust monthly modelling tables from SCMD aggregates."""
import pandas as pd
from config import OUTPUT_DIR, MIN_HISTORY_MONTHS, FORECAST_TARGETS
from data_loader import load_raw_data, filter_series, create_continuous_monthly_series
from feature_engineering import add_time_features, add_target_features, add_intermittent_flag


def build_modeling_dataset(raw=None, min_months=MIN_HISTORY_MONTHS):
    raw = load_raw_data() if raw is None else raw
    eligible = filter_series(raw, min_months=min_months, target="GROSS_POSITIVE_QUANTITY")
    monthly = add_time_features(create_continuous_monthly_series(eligible))
    outputs = {}
    for target in FORECAST_TARGETS:
        featured = add_target_features(monthly, target)
        outputs[target] = add_intermittent_flag(featured, target)
    return monthly, outputs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly, outputs = build_modeling_dataset()
    monthly.to_csv(OUTPUT_DIR / "medicine_trust_monthly_dataset.csv", index=False)
    for target, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / f"modeling_{target.lower()}.csv", index=False)
    print(f"Saved {len(monthly):,} monthly rows across "
          f"{monthly['MEDICINE_HOSPITAL_ID'].nunique():,} medicine-Trust-unit series.")


if __name__ == "__main__":
    main()
