import numpy as np
import pandas as pd
import pytest

from data_loader import _clean_text, load_raw_data, create_continuous_monthly_series
from demand_scenarios import create_abrupt_surge_scenario
from feature_engineering import add_target_features
from inventory_simulation import simulate_inventory
from reliability_monitoring import add_reliability_columns, evaluate_forecast_metrics


def source_rows(tmp_path):
    rows = [
        ["202501", "TR1", "100", "Medicine A", "U1", "tablet", 5, 10],
        ["202501", "TR1", "100", "Medicine A", "U1", "tablet", -2, 4],
        ["202501", "TR1", "100", "Medicine A", "U2", "ml", 3, 6],
        ["202502", "TR1", "100", "Medicine A", "U1", "tablets", -1, 2],
        ["202503", "TR1", "100", "Medicine A", "U1", "tablet", 0, -2],
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


def test_chunk_aggregation_preserves_gross_negative_net_and_unit_identity(tmp_path):
    result = load_raw_data(source_rows(tmp_path), chunksize=2)
    assert len(result) == 4
    u1 = result[result["UNIT_OF_MEASURE_IDENTIFIER"] == "U1"].sort_values("DATE")
    jan, feb = u1.iloc[0], u1.iloc[1]
    assert jan["GROSS_POSITIVE_QUANTITY"] == 5
    assert jan["NEGATIVE_QUANTITY_TOTAL"] == -2
    assert jan["NET_QUANTITY"] == 3
    assert jan["NEGATIVE_QUANTITY_COUNT"] == 1
    assert pd.isna(feb["GROSS_POSITIVE_QUANTITY"])
    assert feb["NEGATIVE_QUANTITY_TOTAL"] == -1
    assert feb["NET_QUANTITY"] == -1


def test_unit_name_variation_does_not_split_series_and_is_reported(tmp_path):
    result = load_raw_data(source_rows(tmp_path), chunksize=1)
    u1 = result[result["UNIT_OF_MEASURE_IDENTIFIER"] == "U1"]
    assert len(u1) == 3
    assert u1["UNIT_NAME_VARIANTS"].eq(2).all()


def test_missing_month_measurements_remain_missing(tmp_path):
    result = load_raw_data(source_rows(tmp_path), chunksize=2)
    monthly = result[result["UNIT_OF_MEASURE_IDENTIFIER"] == "U1"]
    continuous = create_continuous_monthly_series(monthly)
    feb = continuous[continuous["YEAR_MONTH"] == "202502"].iloc[0]
    assert pd.isna(feb["GROSS_POSITIVE_QUANTITY"])
    assert feb["OBSERVATION_MISSING"] == 0  # a negative-only source month was reported


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


def test_negative_movements_are_separate_inflows_not_negative_demand():
    result = simulate_inventory(sample_inventory(), apply_negative_movements_as_returns=True)
    assert result.loc[1, "RETURN_ADJUSTMENT_INFLOW"] == 2
    assert result.loc[1, "SHORTAGE_QUANTITY"] >= 0
    assert result["INVENTORY_SIMULATION_IS_HYPOTHETICAL"].eq(1).all()


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
