import numpy as np
import pandas as pd


GROUP = "MEDICINE_HOSPITAL_ID"


def add_time_features(df):
    df = df.copy()
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    df["QUARTER"] = df["DATE"].dt.quarter
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    return df


def add_lag_features(df, lags=(1, 2, 3)):
    df = df.sort_values([GROUP, "DATE"]).copy()
    for lag in lags:
        df[f"DEMAND_LAG_{lag}"] = df.groupby(GROUP)["DEMAND"].shift(lag)
    return df


def add_rolling_features(df, window=3):
    df = df.sort_values([GROUP, "DATE"]).copy()
    grouped = df.groupby(GROUP)["DEMAND"]
    df["ROLLING_MEAN_3M"] = grouped.transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df["ROLLING_STD_3M"] = grouped.transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).std()
    )
    df["ROLLING_MAX_3M"] = grouped.transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).max()
    )
    return df


def add_intermittent_flag(df, zero_threshold=0.30):
    df = df.copy()
    summary = df.groupby(GROUP)["DEMAND"].agg(
        ZERO_DEMAND_SHARE=lambda x: x.eq(0).sum() / max(x.notna().sum(), 1),
        OBSERVED_MONTHS="count",
    ).reset_index()
    summary["IS_INTERMITTENT"] = (
        summary["ZERO_DEMAND_SHARE"] >= zero_threshold
    ).astype("int8")
    return df.merge(summary, on=GROUP, how="left")


def assign_clinical_priority(df, ven_mapping=None):
    """Only assign VEN from an externally verified mapping; never fabricate it."""
    df = df.copy()
    if ven_mapping is None:
        df["VEN_CLASS"] = "UNCLASSIFIED"
        df["SUBSTITUTABLE"] = pd.NA
        df["CLINICAL_IMPACT_WEIGHT"] = np.nan
        return df

    required = {"VMP_SNOMED_CODE", "VEN_CLASS"}
    if not required.issubset(ven_mapping.columns):
        raise ValueError(f"VEN mapping must contain {sorted(required)}")

    mapping = ven_mapping[["VMP_SNOMED_CODE", "VEN_CLASS"]].drop_duplicates()
    df = df.merge(mapping, on="VMP_SNOMED_CODE", how="left")
    df["VEN_CLASS"] = df["VEN_CLASS"].fillna("UNCLASSIFIED")
    return df
