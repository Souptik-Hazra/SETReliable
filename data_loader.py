import pandas as pd

from config import (
    INPUT_FILE, CHUNK_SIZE, YEAR_MONTH_COL, ODS_CODE_COL,
    MEDICINE_CODE_COL, MEDICINE_NAME_COL, UNIT_ID_COL, UNIT_COL,
    QUANTITY_COL, COST_COL, MIN_HISTORY_MONTHS,
)

REQUIRED_COLUMNS = [
    YEAR_MONTH_COL, ODS_CODE_COL, MEDICINE_CODE_COL, MEDICINE_NAME_COL,
    UNIT_ID_COL, UNIT_COL, QUANTITY_COL, COST_COL,
]
KEYS = ["DATE", ODS_CODE_COL, MEDICINE_CODE_COL, UNIT_ID_COL, UNIT_COL]


def _clean_text(series):
    return series.astype("string").str.strip().replace("", pd.NA)


def load_raw_data(path=INPUT_FILE, chunksize=CHUNK_SIZE):
    """Chunk-read, normalize, and partially aggregate the NHSBSA CSV.

    Negative quantities are retained in SIGNED_NET_QUANTITY and separately
    counted. DEMAND is the sum of positive rows only; this is an explicit
    modelling convention, not an assertion that negative rows are invalid.
    Unit identifier and unit name both remain part of the series key.
    """
    partials = []
    seen_columns = False
    for chunk in pd.read_csv(
        path,
        dtype={YEAR_MONTH_COL: "string", ODS_CODE_COL: "string",
               MEDICINE_CODE_COL: "string", UNIT_ID_COL: "string"},
        chunksize=chunksize,
        low_memory=False,
    ):
        missing = [c for c in REQUIRED_COLUMNS if c not in chunk.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        seen_columns = True

        chunk["DATE"] = pd.to_datetime(
            chunk[YEAR_MONTH_COL].astype("string").str.strip(),
            format="%Y%m", errors="coerce"
        )
        chunk[ODS_CODE_COL] = _clean_text(chunk[ODS_CODE_COL])
        chunk[MEDICINE_CODE_COL] = _clean_text(chunk[MEDICINE_CODE_COL])
        chunk[MEDICINE_NAME_COL] = _clean_text(chunk[MEDICINE_NAME_COL])
        chunk[UNIT_ID_COL] = _clean_text(chunk[UNIT_ID_COL]).fillna("UNKNOWN_UNIT_ID")
        chunk[UNIT_COL] = _clean_text(chunk[UNIT_COL])
        chunk[QUANTITY_COL] = pd.to_numeric(chunk[QUANTITY_COL], errors="coerce")
        chunk[COST_COL] = pd.to_numeric(chunk[COST_COL], errors="coerce")
        chunk = chunk.dropna(subset=KEYS + [QUANTITY_COL])

        chunk["POSITIVE_DEMAND"] = chunk[QUANTITY_COL].where(chunk[QUANTITY_COL] >= 0)
        chunk["NEGATIVE_QUANTITY_COUNT"] = (chunk[QUANTITY_COL] < 0).astype("int32")
        chunk["NEGATIVE_COST_COUNT"] = (chunk[COST_COL] < 0).astype("int32")
        chunk["VALID_COST"] = chunk[COST_COL].where(chunk[COST_COL] >= 0)
        part = chunk.groupby(KEYS, as_index=False, observed=True, dropna=False).agg(
            DEMAND=("POSITIVE_DEMAND", lambda x: x.sum(min_count=1)),
            SIGNED_NET_QUANTITY=(QUANTITY_COL, "sum"),
            NEGATIVE_QUANTITY_COUNT=("NEGATIVE_QUANTITY_COUNT", "sum"),
            NEGATIVE_COST_COUNT=("NEGATIVE_COST_COUNT", "sum"),
            INDICATIVE_COST_TOTAL=("VALID_COST", lambda x: x.sum(min_count=1)),
            COST_RECORDS=("VALID_COST", "count"),
            **{
                MEDICINE_NAME_COL: (MEDICINE_NAME_COL, "first"),
            },
        )
        partials.append(part)

    if not seen_columns or not partials:
        raise ValueError(f"No valid records found in {path}")

    # Only aggregated chunk results are concatenated, not the 10m raw rows.
    df = pd.concat(partials, ignore_index=True)
    df = df.groupby(KEYS, as_index=False, observed=True, dropna=False).agg(
        DEMAND=("DEMAND", lambda x: x.sum(min_count=1)),
        SIGNED_NET_QUANTITY=("SIGNED_NET_QUANTITY", "sum"),
        NEGATIVE_QUANTITY_COUNT=("NEGATIVE_QUANTITY_COUNT", "sum"),
        NEGATIVE_COST_COUNT=("NEGATIVE_COST_COUNT", "sum"),
        INDICATIVE_COST_TOTAL=("INDICATIVE_COST_TOTAL", lambda x: x.sum(min_count=1)),
        COST_RECORDS=("COST_RECORDS", "sum"),
        **{MEDICINE_NAME_COL: (MEDICINE_NAME_COL, "first")},
    )
    df["COST_PER_RECORD"] = (
        df["INDICATIVE_COST_TOTAL"] / df["COST_RECORDS"].replace(0, pd.NA)
    )
    df["YEAR_MONTH"] = df["DATE"].dt.strftime("%Y%m")
    # Length-prefixed components avoid ambiguous delimiter-concatenation keys.
    components = [df[ODS_CODE_COL], df[MEDICINE_CODE_COL], df[UNIT_ID_COL], df[UNIT_COL]]
    df["MEDICINE_HOSPITAL_ID"] = components[0].str.len().astype(str) + ":" + components[0] + "|" + (
        components[1].str.len().astype(str) + ":" + components[1]
    ) + "|" + (components[2].fillna("").str.len().astype(str) + ":" + components[2].fillna("")) + "|" + (
        components[3].str.len().astype(str) + ":" + components[3]
    )
    return df.sort_values(["MEDICINE_HOSPITAL_ID", "DATE"]).reset_index(drop=True)


def filter_series(df, min_months=MIN_HISTORY_MONTHS):
    valid_counts = df.loc[df["DEMAND"].notna()].groupby(
        "MEDICINE_HOSPITAL_ID"
    )["DATE"].nunique()
    valid_ids = valid_counts[valid_counts >= min_months].index
    return df[df["MEDICINE_HOSPITAL_ID"].isin(valid_ids)].copy()


def create_continuous_monthly_series(df):
    """Reindex to calendar months; unreported months remain missing, not zero."""
    frames = []
    metadata = [ODS_CODE_COL, MEDICINE_CODE_COL, MEDICINE_NAME_COL, UNIT_ID_COL, UNIT_COL]
    for series_id, group in df.groupby("MEDICINE_HOSPITAL_ID", sort=False):
        group = group.sort_values("DATE").set_index("DATE")
        index = pd.date_range(group.index.min(), group.index.max(), freq="MS")
        group = group.reindex(index)
        group.index.name = "DATE"
        group = group.reset_index()
        group["MEDICINE_HOSPITAL_ID"] = series_id
        for col in metadata:
            group[col] = group[col].ffill().bfill()
        group["YEAR_MONTH"] = group["DATE"].dt.strftime("%Y%m")
        group["OBSERVATION_MISSING"] = group["DEMAND"].isna().astype("int8")
        frames.append(group)
    if not frames:
        raise ValueError("No eligible medicine-Trust-unit series remain.")
    return pd.concat(frames, ignore_index=True)
