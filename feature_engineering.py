import numpy as np
import pandas as pd

GROUP = "MEDICINE_HOSPITAL_ID"
DEFAULT_LAGS = (1, 2, 3, 6, 12)


def add_time_features(df):
    out = df.copy()
    out["YEAR"] = out["DATE"].dt.year
    out["MONTH"] = out["DATE"].dt.month
    out["QUARTER"] = out["DATE"].dt.quarter
    out["MONTH_SIN"] = np.sin(2 * np.pi * out["MONTH"] / 12)
    out["MONTH_COS"] = np.cos(2 * np.pi * out["MONTH"] / 12)
    return out


def add_target_features(df, target, lags=DEFAULT_LAGS, windows=(3, 6)):
    """Create strictly past-only features; missing source months stay missing."""
    if target not in df:
        raise ValueError(f"Target column not found: {target}")
    out = df.sort_values([GROUP, "DATE"]).copy()
    grouped = out.groupby(GROUP, sort=False)[target]
    for lag in lags:
        out[f"{target}_LAG_{lag}"] = grouped.shift(lag)
    for window in windows:
        prior = grouped.shift(1)
        out[f"{target}_ROLLING_MEAN_{window}M"] = prior.groupby(out[GROUP]).transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        out[f"{target}_ROLLING_STD_{window}M"] = prior.groupby(out[GROUP]).transform(
            lambda x: x.rolling(window, min_periods=2).std())
        out[f"{target}_ROLLING_MAX_{window}M"] = prior.groupby(out[GROUP]).transform(
            lambda x: x.rolling(window, min_periods=1).max())
    return out


def add_intermittent_flag(df, target, zero_threshold=0.30):
    if target not in df:
        raise ValueError(f"Target column not found: {target}")
    out = df.copy()
    summary = out.groupby(GROUP)[target].agg(
        ZERO_SHARE=lambda x: x.eq(0).sum() / max(x.notna().sum(), 1),
        OBSERVED_MONTHS="count",
    ).reset_index()
    summary["IS_INTERMITTENT"] = (summary["ZERO_SHARE"] >= zero_threshold).astype("int8")
    summary = summary.rename(columns={
        "ZERO_SHARE": f"{target}_ZERO_SHARE",
        "OBSERVED_MONTHS": f"{target}_OBSERVED_MONTHS",
        "IS_INTERMITTENT": f"{target}_IS_INTERMITTENT",
    })
    return out.merge(summary, on=GROUP, how="left")


def assign_clinical_priority(df, ven_mapping=None):
    """Never infer clinical criticality from a medicine-code hash."""
    out = df.copy()
    if ven_mapping is None:
        out["VEN_CLASS"] = "UNCLASSIFIED"
        out["SUBSTITUTABLE"] = pd.NA
        out["CLINICAL_IMPACT_WEIGHT"] = np.nan
        return out
    required = {"VMP_SNOMED_CODE", "VEN_CLASS"}
    if not required.issubset(ven_mapping.columns):
        raise ValueError(f"VEN mapping must contain {sorted(required)}")
    mapping = ven_mapping[[c for c in (
        "VMP_SNOMED_CODE", "VEN_CLASS", "SUBSTITUTABLE", "CLINICAL_IMPACT_WEIGHT"
    ) if c in ven_mapping.columns]].drop_duplicates()
    if mapping["VMP_SNOMED_CODE"].duplicated().any():
        raise ValueError("VEN mapping must contain at most one row per VMP_SNOMED_CODE")
    return out.merge(mapping, on="VMP_SNOMED_CODE", how="left").assign(
        VEN_CLASS=lambda x: x["VEN_CLASS"].fillna("UNCLASSIFIED"))
