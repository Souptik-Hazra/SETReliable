import numpy as np
import pandas as pd
import pytest

from data_loader import _clean_text
from inventory_simulation import calculate_initial_inventory, simulate_inventory
from reliability_monitoring import add_reliability_columns


def sample_frame():
    return pd.DataFrame({
        "MEDICINE_HOSPITAL_ID": ["series-a"] * 5,
        "DATE": pd.date_range("2025-01-01", periods=5, freq="MS"),
        "DEMAND": [10.0, 12.0, np.nan, 8.0, 11.0],
        "DEMAND_SCENARIO": [10.0, 12.0, np.nan, 8.0, 11.0],
        "LEAD_TIME_MONTHS": [1] * 5,
    })


def test_text_cleaning_keeps_missing_values():
    result = _clean_text(pd.Series([" A ", "", None]))
    assert result.iloc[0] == "A"
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])


def test_initial_inventory_does_not_use_future_demand():
    frame = sample_frame()
    initialized = calculate_initial_inventory(frame, initial_inventory=7)
    assert initialized["INITIAL_INVENTORY"].eq(7).all()


def test_missing_month_is_not_zero_demand_or_consumption():
    frame = sample_frame()
    frame["INITIAL_INVENTORY"] = 20.0
    simulated = simulate_inventory(frame)
    missing_row = simulated.loc[simulated["DEMAND_SCENARIO"].isna()].iloc[0]
    assert missing_row["SHORTAGE_QUANTITY"] == 0
    assert missing_row["INVENTORY_ON_HAND"] == missing_row["AVAILABLE_SUPPLY"]


def test_simulation_never_creates_negative_stock_or_order():
    frame = sample_frame()
    frame["INITIAL_INVENTORY"] = 0.0
    simulated = simulate_inventory(frame)
    for col in ["INVENTORY_ON_HAND", "ORDER_QUANTITY", "SHORTAGE_QUANTITY"]:
        assert (simulated[col] >= 0).all()


def test_reliability_uses_prior_errors_for_interval():
    frame = sample_frame()
    result = add_reliability_columns(frame, window=2)
    assert "PREDICTION_INTERVAL_LOWER" in result
    assert "PREDICTION_INTERVAL_UPPER" in result
    assert "FORECAST_P10" not in result
    assert pd.isna(result.loc[0, "RELIABILITY_FORECAST"])


def test_invalid_reorder_quantile_rejected():
    frame = sample_frame()
    frame["INITIAL_INVENTORY"] = 0.0
    with pytest.raises(ValueError):
        simulate_inventory(frame, reorder_quantile=1.0)

def test_chunked_loader_aggregates_across_chunks_and_keeps_units(tmp_path):
    from data_loader import load_raw_data

    rows = [
        ["202501", "TR1", "100", "Medicine A", "U1", "tablet", 5, 10],
        ["202501", "TR1", "100", "Medicine A", "U1", "tablet", -2, 4],
        ["202501", "TR1", "100", "Medicine A", "U2", "ml", 3, 6],
        ["202502", "TR1", "100", "Medicine A", "U1", "tablet", -1, 2],
    ]
    columns = [
        "YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE", "VMP_PRODUCT_NAME",
        "UNIT_OF_MEASURE_IDENTIFIER", "UNIT_OF_MEASURE_NAME",
        "TOTAL_QUANITY_IN_VMP_UNIT", "INDICATIVE_COST",
    ]
    path = tmp_path / "sample.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)

    result = load_raw_data(path, chunksize=2)
    assert len(result) == 3
    tablet = result[result["UNIT_OF_MEASURE_IDENTIFIER"] == "U1"]
    jan = tablet[tablet["YEAR_MONTH"] == "202501"].iloc[0]
    feb = tablet[tablet["YEAR_MONTH"] == "202502"].iloc[0]
    assert jan["DEMAND"] == 5
    assert jan["SIGNED_NET_QUANTITY"] == 3
    assert jan["NEGATIVE_QUANTITY_COUNT"] == 1
    assert pd.isna(feb["DEMAND"])
