import numpy as np
import pandas as pd


def create_stable_scenario(df):
    df = df.copy()
    df["SCENARIO_ID"] = "STABLE"
    df["DEMAND_REGIME"] = "STABLE"
    df["DEMAND_SCENARIO"] = df["DEMAND"]
    return df


def create_abrupt_surge_scenario(
    df,
    surge_factor=2.0,
    surge_months=3,
):
    df = df.copy()
    df = df.sort_values(
        by=["MEDICINE_HOSPITAL_ID", "DATE"]
    )

    df["DEMAND_SCENARIO"] = df["DEMAND"]
    df["SCENARIO_ID"] = f"ABRUPT_SURGE_{surge_factor}"
    df["DEMAND_REGIME"] = "STABLE"

    for series_id, indexes in df.groupby("MEDICINE_HOSPITAL_ID").groups.items():
        indexes = list(indexes)
        n = len(indexes)

        if n <= surge_months + 6:
            continue

        start_position = int(n * 0.60)
        end_position = min(start_position + surge_months, n)

        surge_indexes = indexes[start_position:end_position]

        df.loc[surge_indexes, "DEMAND_SCENARIO"] = (
            df.loc[surge_indexes, "DEMAND"] * surge_factor
        )

        df.loc[surge_indexes, "DEMAND_REGIME"] = "ABRUPT_SURGE"

    return df


def create_gradual_drift_scenario(
    df,
    total_increase=0.50,
    drift_months=6,
):
    df = df.copy()
    df = df.sort_values(
        by=["MEDICINE_HOSPITAL_ID", "DATE"]
    )

    df["DEMAND_SCENARIO"] = df["DEMAND"]
    df["SCENARIO_ID"] = f"GRADUAL_DRIFT_{total_increase}"
    df["DEMAND_REGIME"] = "STABLE"

    for series_id, indexes in df.groupby("MEDICINE_HOSPITAL_ID").groups.items():
        indexes = list(indexes)
        n = len(indexes)

        if n <= drift_months + 6:
            continue

        start_position = int(n * 0.60)
        end_position = min(start_position + drift_months, n)

        drift_indexes = indexes[start_position:end_position]

        relative_progress = np.linspace(
            0,
            1,
            len(drift_indexes),
        )

        multiplier = 1 + total_increase * relative_progress

        df.loc[drift_indexes, "DEMAND_SCENARIO"] = (
            df.loc[drift_indexes, "DEMAND"] * multiplier
        )

        df.loc[drift_indexes, "DEMAND_REGIME"] = "GRADUAL_DRIFT"

    return df


def create_volatility_scenario(
    df,
    volatility_multiplier=2.0,
):
    df = df.copy()
    df = df.sort_values(
        by=["MEDICINE_HOSPITAL_ID", "DATE"]
    )

    df["DEMAND_SCENARIO"] = df["DEMAND"]
    df["SCENARIO_ID"] = f"VOLATILITY_{volatility_multiplier}"
    df["DEMAND_REGIME"] = "STABLE"

    for series_id, group in df.groupby("MEDICINE_HOSPITAL_ID"):
        indexes = group.index
        demand = group["DEMAND"].to_numpy()

        if len(demand) < 12:
            continue

        mean_demand = demand.mean()

        modified_demand = (
            mean_demand
            + volatility_multiplier * (demand - mean_demand)
        )

        modified_demand = np.maximum(modified_demand, 0)

        df.loc[indexes, "DEMAND_SCENARIO"] = modified_demand

        change_start = int(len(demand) * 0.60)

        df.loc[indexes[change_start:], "DEMAND_REGIME"] = (
            "VOLATILITY_INCREASE"
        )

    return df