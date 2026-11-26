import numpy as np
import pandas as pd
import pytest

from data_loader import _clean_text, load_raw_data, create_continuous_monthly_series
from demand_scenarios import create_abrupt_surge_scenario
from feature_engineering import add_target_features
from inventory_simulation import simulate_inventory
from reliability_monitoring import add_reliability_columns, evaluate_forecast_metrics
from data_quality_analysis import audit


def source_rows(tmp_path):
    rows = [
        ["202501", "TR1", "100", "Medicine A", "U1", "tablet", 5, 10],
        ["202501", "TR1", "100", "Medicine A", "U1", "tablet", -2, 4],
        ["202501", "TR1", "100", "Medicine A", "U2", "ml", 3, 6],
        ["202502", "TR1", "100", "Medicine A", "U1", "tablets", -1, 2],
        ["202503", "TR1", "100", "Medicine A", "U1", "tablet", 0, -2],
        ["202505", "TR1", "100", "Medicine A", "U1", "tablet", 4, 8],
    ]
    columns = [
        "YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE", "VMP_PRODUCT_NAME",
        "UNIT_OF_MEASURE_IDENTIFIER", "UNIT_OF_MEASURE_NAME",
        "TOTAL_QUANITY_IN_VMP_UNIT", "INDICATIVE_COST",
    ]
    path = tmp_path / "sample.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_text_cleaning():
    result = _clean_text(pd.Series([" A ", "", None]))
    assert result.iloc[0] == "A"
    assert pd.isna(result.iloc[1]) and pd.isna(result.iloc[2])


def test_chunk_aggregation_distinguishes_zero_from_unobserved(tmp_path):
    result = load_raw_data(source_rows(tmp_path), memory_limit="256MB")
    u1 = result[result["UNIT_OF_MEASURE_IDENTIFIER"] == "U1"].sort_values("DATE")
    jan, feb, mar = u1.iloc[0], u1.iloc[1], u1.iloc[2]
    assert jan["GROSS_POSITIVE_QUANTITY"] == 5
    assert jan["NEGATIVE_QUANTITY_TOTAL"] == -2
    assert jan["NET_QUANTITY"] == 3
    assert feb["GROSS_POSITIVE_QUANTITY"] == 0
    assert feb["NEGATIVE_QUANTITY_TOTAL"] == -1
    assert feb["NET_QUANTITY"] == -1
    assert mar["GROSS_POSITIVE_QUANTITY"] == 0
    assert mar["NEGATIVE_QUANTITY_TOTAL"] == 0


def test_missing_calendar_month_stays_missing(tmp_path):
    result = load_raw_data(source_rows(tmp_path))
    u1 = result[result["UNIT_OF_MEASURE_IDENTIFIER"] == "U1"]
    continuous = create_continuous_monthly_series(u1)
    april = continuous[continuous["YEAR_MONTH"] == "202504"].iloc[0]
    feb = continuous[continuous["YEAR_MONTH"] == "202502"].iloc[0]
    assert pd.isna(april["GROSS_POSITIVE_QUANTITY"])
    assert april["OBSERVATION_MISSING"] == 1
    assert feb["GROSS_POSITIVE_QUANTITY"] == 0
    assert feb["OBSERVATION_MISSING"] == 0


def test_missing_unit_identifier_uses_same_sentinel(tmp_path):
    path = source_rows(tmp_path)
    frame = pd.read_csv(path, dtype="string")
    frame.loc[0, "UNIT_OF_MEASURE_IDENTIFIER"] = pd.NA
    frame.to_csv(path, index=False)
    result = load_raw_data(path)
    assert "UNKNOWN_UNIT_ID" in set(result["UNIT_OF_MEASURE_IDENTIFIER"])


def sample_inventory():
    return pd.DataFrame({
        "MEDICINE_HOSPITAL_ID": ["s"] * 4,
        "DATE": pd.date_range("2025-01-01", periods=4, freq="MS"),
        "GROSS_POSITIVE_QUANTITY": [10.0, 12.0, np.nan, 8.0],
        "NEGATIVE_QUANTITY_TOTAL": [0.0, -2.0, np.nan, 0.0],
        "LEAD_TIME_MONTHS": [1] * 4,
        "INITIAL_INVENTORY": [20.0] * 4,
        "SCENARIO_QUANTITY": [10.0, 12.0, np.nan, 8.0],
    })


def test_negative_movements_are_not_returns_by_default():
    result = simulate_inventory(sample_inventory())
    assert result.loc[1, "RETURN_ADJUSTMENT_INFLOW"] == 0
    assert result["NEGATIVE_MOVEMENTS_TREATED_AS_RETURNS"].eq(0).all()
    assert result["STOCKOUTS_ARE_OBSERVED"].eq(0).all()


def test_negative_movements_can_be_enabled_only_as_sensitivity_assumption():
    result = simulate_inventory(sample_inventory(), apply_negative_movements_as_returns=True)
    assert result.loc[1, "RETURN_ADJUSTMENT_INFLOW"] == 2


def test_missing_month_does_not_consume_stock():
    result = simulate_inventory(sample_inventory())
    missing = result.loc[2]
    assert missing["SHORTAGE_QUANTITY"] == 0
    assert missing["INVENTORY_ON_HAND"] == missing["AVAILABLE_SUPPLY"]


def test_invalid_reorder_quantile_rejected():
    with pytest.raises(ValueError):
        simulate_inventory(sample_inventory(), reorder_quantile=1)


def test_forecast_uses_prior_month_and_not_current_month():
    frame = pd.DataFrame({
        "MEDICINE_HOSPITAL_ID": ["s"] * 4,
        "DATE": pd.date_range("2025-01-01", periods=4, freq="MS"),
        "GROSS_POSITIVE_QUANTITY": [10.0, 12.0, 15.0, 8.0],
    })
    featured = add_target_features(frame, "GROSS_POSITIVE_QUANTITY")
    result = add_reliability_columns(featured, target="GROSS_POSITIVE_QUANTITY", window=2)
    assert pd.isna(result.loc[0, "BASELINE_FORECAST"])
    assert result.loc[2, "BASELINE_FORECAST"] == 12.0
    metrics = evaluate_forecast_metrics(result)
    assert metrics["n"] == 3
    assert metrics["mae"] >= 0


def test_scenario_preserves_missing_observations():
    result = create_abrupt_surge_scenario(sample_inventory(), surge_months=2)
    assert pd.isna(result.loc[2, "SCENARIO_QUANTITY"])


def test_extended_lags_and_rolling_features_are_past_only():
    frame = pd.DataFrame({
        "MEDICINE_HOSPITAL_ID": ["s"] * 15,
        "DATE": pd.date_range("2024-01-01", periods=15, freq="MS"),
        "GROSS_POSITIVE_QUANTITY": np.arange(1, 16, dtype=float),
    })
    result = add_target_features(frame, "GROSS_POSITIVE_QUANTITY")
    assert result.loc[12, "GROSS_POSITIVE_QUANTITY_LAG_12"] == 1
    assert result.loc[3, "GROSS_POSITIVE_QUANTITY_ROLLING_MEAN_3M"] == 2


def test_drift_monitors_are_available_and_causal():
    from reliability_monitoring import _cusum_flags, _adwin_like_flags
    errors = np.array([1.0] * 12 + [20.0] * 8)
    cusum = _cusum_flags(errors, window=4)
    adaptive = _adwin_like_flags(errors, min_window=3)
    assert len(cusum) == len(errors)
    assert len(adaptive) == len(errors)


def test_duplicate_audit_detects_within_and_across_chunks(tmp_path):
    path = source_rows(tmp_path)
    frame = pd.read_csv(path, dtype="string")
    duplicate = frame.iloc[[0]].copy()
    data = pd.concat([frame, duplicate, frame.iloc[[1]]], ignore_index=True)
    data.to_csv(path, index=False)
    result = audit(path, chunksize=2)
    assert result["exact_duplicate_rows"] >= 2
    assert result["duplicate_series_month_keys"] >= 3


def test_loader_rejects_missing_required_schema(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"YEAR_MONTH": ["202501"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_raw_data(path)


def test_audit_rejects_missing_required_schema(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"YEAR_MONTH": ["202501"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        audit(path)


def test_reliability_score_is_finite_and_warmup_is_explicit():
    frame = pd.DataFrame({
        "MEDICINE_HOSPITAL_ID": ["s"] * 5,
        "DATE": pd.date_range("2025-01-01", periods=5, freq="MS"),
        "GROSS_POSITIVE_QUANTITY": [10.0, 12.0, 15.0, 8.0, 11.0],
    })
    result = add_reliability_columns(frame, target="GROSS_POSITIVE_QUANTITY", window=2)
    assert np.isfinite(result["RELIABILITY_SCORE"]).all()
    assert result.loc[0, "RELIABILITY_SCORE_AVAILABLE"] == 0
    assert result.loc[0, "RISK_ALERT"] == 0
    assert result.loc[1:, "RELIABILITY_SCORE_AVAILABLE"].eq(1).all()


def test_unit_name_audit_is_unit_aware_and_global(tmp_path):
    from analyze_unit_consistency import analyze
    path = source_rows(tmp_path)
    frame = pd.read_csv(path, dtype="string")
    frame.loc[frame["UNIT_OF_MEASURE_IDENTIFIER"] == "U1", "UNIT_OF_MEASURE_NAME"] = "tablet"
    frame.loc[frame["YEAR_MONTH"] == "202502", "UNIT_OF_MEASURE_NAME"] = "tablets"
    frame["ODS_CODE"] = " " + frame["ODS_CODE"] + " "
    frame.to_csv(path, index=False)
    result = analyze(path, chunksize=2)
    u1 = result[result["UNIT_OF_MEASURE_IDENTIFIER"] == "U1"].iloc[0]
    u2 = result[result["UNIT_OF_MEASURE_IDENTIFIER"] == "U2"].iloc[0]
    assert u1["UNIT_NAME_VARIANTS"] == 2
    assert u1["UNIT_NAMES"] == "tablet | tablets"
    assert u2["UNIT_NAME_VARIANTS"] == 1
