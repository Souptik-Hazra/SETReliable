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
SERIES_KEYS = ["DATE", ODS_CODE_COL, MEDICINE_CODE_COL, UNIT_ID_COL]


def _clean_text(series):
    return series.astype("string").str.strip().replace("", pd.NA)


def load_raw_data(path=INPUT_FILE, chunksize=CHUNK_SIZE):
    """Aggregate the pre-June-2026 SCMD schema without discarding signed movements."""
    partials = []
    quality = {
        "source_rows": 0, "invalid_month_rows": 0, "missing_identity_rows": 0,
        "invalid_quantity_rows": 0, "missing_cost_rows": 0,
        "negative_quantity_rows": 0, "zero_quantity_rows": 0, "negative_cost_rows": 0,
    }
    for chunk in pd.read_csv(
        path,
        dtype={YEAR_MONTH_COL: "string", ODS_CODE_COL: "string",
               MEDICINE_CODE_COL: "string", UNIT_ID_COL: "string"},
        chunksize=chunksize, low_memory=False,
    ):
        missing = sorted(set(REQUIRED_COLUMNS) - set(chunk.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        quality["source_rows"] += len(chunk)
        chunk["DATE"] = pd.to_datetime(
            chunk[YEAR_MONTH_COL].astype("string").str.strip(),
            format="%Y%m", errors="coerce",
        )
        quantity = pd.to_numeric(chunk[QUANTITY_COL], errors="coerce")
        cost = pd.to_numeric(chunk[COST_COL], errors="coerce")
        quality["invalid_month_rows"] += int(chunk["DATE"].isna().sum())
        quality["missing_cost_rows"] += int(cost.isna().sum())
        quality["invalid_quantity_rows"] += int(quantity.isna().sum())
        quality["negative_quantity_rows"] += int((quantity < 0).sum())
        quality["zero_quantity_rows"] += int((quantity == 0).sum())
        quality["negative_cost_rows"] += int((cost < 0).sum())
        for col in (ODS_CODE_COL, MEDICINE_CODE_COL, MEDICINE_NAME_COL, UNIT_ID_COL, UNIT_COL):
            chunk[col] = _clean_text(chunk[col])
        chunk[UNIT_ID_COL] = chunk[UNIT_ID_COL].fillna("UNKNOWN_UNIT_ID")
        quality["missing_identity_rows"] += int(
            chunk[[ODS_CODE_COL, MEDICINE_CODE_COL]].isna().any(axis=1).sum()
        )
        chunk[QUANTITY_COL], chunk[COST_COL] = quantity, cost
        chunk = chunk.dropna(subset=["DATE", ODS_CODE_COL, MEDICINE_CODE_COL, QUANTITY_COL])
        chunk["POSITIVE"] = chunk[QUANTITY_COL].where(chunk[QUANTITY_COL] > 0)
        chunk["NEGATIVE"] = chunk[QUANTITY_COL].where(chunk[QUANTITY_COL] < 0)
        chunk["NEGATIVE_QUANTITY_COUNT"] = (chunk[QUANTITY_COL] < 0).astype("int32")
        chunk["ZERO_QUANTITY_COUNT"] = (chunk[QUANTITY_COL] == 0).astype("int32")
        chunk["NEGATIVE_COST_COUNT"] = (chunk[COST_COL] < 0).astype("int32")
        part = chunk.groupby(SERIES_KEYS, as_index=False, observed=True, dropna=False).agg(
            GROSS_POSITIVE_QUANTITY=("POSITIVE", lambda x: x.sum(min_count=1)),
            NEGATIVE_QUANTITY_TOTAL=("NEGATIVE", lambda x: x.sum(min_count=1)),
            NET_QUANTITY=(QUANTITY_COL, "sum"),
            NEGATIVE_QUANTITY_COUNT=("NEGATIVE_QUANTITY_COUNT", "sum"),
            ZERO_QUANTITY_COUNT=("ZERO_QUANTITY_COUNT", "sum"),
            NEGATIVE_COST_COUNT=("NEGATIVE_COST_COUNT", "sum"),
            INDICATIVE_COST_TOTAL=(COST_COL, lambda x: x.sum(min_count=1)),
            COST_RECORDS=(COST_COL, "count"),
            SOURCE_RECORD_COUNT=(QUANTITY_COL, "size"),
            VMP_PRODUCT_NAME=(MEDICINE_NAME_COL, "first"),
            UNIT_NAME_SET=(UNIT_COL, lambda x: tuple(sorted(set(x.dropna().astype(str))))),
        )
        partials.append(part)

    if not partials:
        raise ValueError(f"No valid records found in {path}")
    data = pd.concat(partials, ignore_index=True)
    data = data.groupby(SERIES_KEYS, as_index=False, observed=True, dropna=False).agg(
        GROSS_POSITIVE_QUANTITY=("GROSS_POSITIVE_QUANTITY", lambda x: x.sum(min_count=1)),
        NEGATIVE_QUANTITY_TOTAL=("NEGATIVE_QUANTITY_TOTAL", lambda x: x.sum(min_count=1)),
        NET_QUANTITY=("NET_QUANTITY", "sum"),
        NEGATIVE_QUANTITY_COUNT=("NEGATIVE_QUANTITY_COUNT", "sum"),
        ZERO_QUANTITY_COUNT=("ZERO_QUANTITY_COUNT", "sum"),
        NEGATIVE_COST_COUNT=("NEGATIVE_COST_COUNT", "sum"),
        INDICATIVE_COST_TOTAL=("INDICATIVE_COST_TOTAL", lambda x: x.sum(min_count=1)),
        COST_RECORDS=("COST_RECORDS", "sum"),
        SOURCE_RECORD_COUNT=("SOURCE_RECORD_COUNT", "sum"),
        VMP_PRODUCT_NAME=("VMP_PRODUCT_NAME", "first"),
        UNIT_NAME_SET=("UNIT_NAME_SET", lambda values: tuple(sorted(
            set().union(*(set(value) for value in values))
        ))),
    )
    data["UNIT_OF_MEASURE_NAME"] = data["UNIT_NAME_SET"].map(
        lambda names: names[0] if names else pd.NA
    )
    data["COST_PER_RECORD"] = (
        data["INDICATIVE_COST_TOTAL"] / data["COST_RECORDS"].replace(0, pd.NA)
    )
    data["YEAR_MONTH"] = data["DATE"].dt.strftime("%Y%m")
    components = [data[ODS_CODE_COL], data[MEDICINE_CODE_COL], data[UNIT_ID_COL]]
    data["MEDICINE_HOSPITAL_ID"] = (
        components[0].str.len().astype(str) + ":" + components[0] + "|" +
        components[1].str.len().astype(str) + ":" + components[1] + "|" +
        components[2].str.len().astype(str) + ":" + components[2]
    )
    # Count distinct unit names across the complete Trust-medicine-unit series,
    # including names appearing in different CSV chunks and different months.
    name_counts = data.groupby("MEDICINE_HOSPITAL_ID")["UNIT_NAME_SET"].agg(
        lambda values: len(set().union(*(set(value) for value in values)))
    )
    data["UNIT_NAME_VARIANTS"] = data["MEDICINE_HOSPITAL_ID"].map(name_counts).astype("int16")
    quality["aggregated_monthly_rows"] = len(data)
    quality["unit_name_conflict_series"] = int((name_counts > 1).sum())
    data = data.sort_values(["MEDICINE_HOSPITAL_ID", "DATE"]).reset_index(drop=True)
    data.attrs["quality_report"] = quality
    return data


def filter_series(df, min_months=MIN_HISTORY_MONTHS, target="GROSS_POSITIVE_QUANTITY"):
    if target not in df:
        raise ValueError(f"Unknown series filter target: {target}")
    counts = df.loc[df[target].notna()].groupby("MEDICINE_HOSPITAL_ID")[target].count()
    valid_ids = counts[counts >= min_months].index
    return df[df["MEDICINE_HOSPITAL_ID"].isin(valid_ids)].copy()


def create_continuous_monthly_series(df):
    """Reindex to calendar months; unreported measurements remain missing."""
    frames = []
    metadata = [ODS_CODE_COL, MEDICINE_CODE_COL, MEDICINE_NAME_COL, UNIT_ID_COL, UNIT_COL]
    for series_id, source in df.groupby("MEDICINE_HOSPITAL_ID", sort=False):
        group = source.sort_values("DATE").set_index("DATE")
        group = group.reindex(pd.date_range(group.index.min(), group.index.max(), freq="MS"))
        group.index.name = "DATE"
        group = group.reset_index()
        group["MEDICINE_HOSPITAL_ID"] = series_id
        for col in metadata:
            if col in group:
                group[col] = group[col].ffill().bfill()
        group["YEAR_MONTH"] = group["DATE"].dt.strftime("%Y%m")
        group["OBSERVATION_MISSING"] = group["SOURCE_RECORD_COUNT"].isna().astype("int8")
        for col in ("NEGATIVE_QUANTITY_COUNT", "ZERO_QUANTITY_COUNT", "NEGATIVE_COST_COUNT",
                    "SOURCE_RECORD_COUNT", "COST_RECORDS"):
            group[col] = group[col].fillna(0)
        frames.append(group)
    if not frames:
        raise ValueError("No eligible medicine-Trust-unit series remain.")
    return pd.concat(frames, ignore_index=True)
