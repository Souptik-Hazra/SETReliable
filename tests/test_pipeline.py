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
