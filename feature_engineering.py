import numpy as np
import pandas as pd


def add_time_features(df):
    df = df.copy()

    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    df["QUARTER"] = df["DATE"].dt.quarter

    return df


def add_lag_features(df, lags=(1, 2, 3)):
    df = df.copy()
    df = df.sort_values(
        by=["MEDICINE_HOSPITAL_ID", "DATE"]
    )

    for lag in lags:
        df[f"DEMAND_LAG_{lag}"] = (
            df.groupby("MEDICINE_HOSPITAL_ID")["DEMAND"]
            .shift(lag)
        )

    return df


def add_rolling_features(df, window=3):
    df = df.copy()
    df = df.sort_values(
        by=["MEDICINE_HOSPITAL_ID", "DATE"]
    )

    grouped = df.groupby("MEDICINE_HOSPITAL_ID")["DEMAND"]

    df["ROLLING_MEAN_3M"] = (
        grouped
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    df["ROLLING_STD_3M"] = (
        grouped
        .transform(lambda x: x.shift(1).rolling(window, min_periods=2).std())
    )

    df["ROLLING_MAX_3M"] = (
        grouped
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
    )

    return df


def add_intermittent_flag(df, zero_threshold=0.30):
    df = df.copy()

    zero_share = (
        df.groupby("MEDICINE_HOSPITAL_ID")["DEMAND"]
        .apply(lambda x: (x == 0).mean())
        .reset_index(name="ZERO_DEMAND_SHARE")
    )

    zero_share["IS_INTERMITTENT"] = (
        zero_share["ZERO_DEMAND_SHARE"] >= zero_threshold
    ).astype(int)

    df = df.merge(
        zero_share,
        on="MEDICINE_HOSPITAL_ID",
        how="left",
    )

    return df


def assign_clinical_priority(df):
    """
    Placeholder assignment.

    Replace this with a real medicine-level VEN mapping if available.
    For initial experimentation, assign priority using a deterministic hash
    so that results are reproducible.
    """

    df = df.copy()

    unique_medicines = (
        df[["VMP_SNOMED_CODE"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    unique_medicines["HASH_VALUE"] = (
        unique_medicines["VMP_SNOMED_CODE"]
        .astype(str)
        .apply(lambda x: abs(hash(x)) % 100)
    )

    def ven_class(hash_value):
        if hash_value < 20:
            return "V"
        elif hash_value < 60:
            return "E"
        return "N"

    unique_medicines["VEN_CLASS"] = unique_medicines["HASH_VALUE"].apply(
        ven_class
    )

    def substitutable(hash_value):
        return int(hash_value >= 40)

    unique_medicines["SUBSTITUTABLE"] = unique_medicines["HASH_VALUE"].apply(
        substitutable
    )

    weight_map = {
        ("V", 0): 1.00,
        ("V", 1): 0.80,
        ("E", 0): 0.60,
        ("E", 1): 0.40,
        ("N", 0): 0.20,
        ("N", 1): 0.10,
    }

    unique_medicines["CLINICAL_IMPACT_WEIGHT"] = unique_medicines.apply(
        lambda row: weight_map[
            (row["VEN_CLASS"], row["SUBSTITUTABLE"])
        ],
        axis=1,
    )

    df = df.merge(
        unique_medicines[
            [
                "VMP_SNOMED_CODE",
                "VEN_CLASS",
                "SUBSTITUTABLE",
                "CLINICAL_IMPACT_WEIGHT",
            ]
        ],
        on="VMP_SNOMED_CODE",
        how="left",
    )

    return df