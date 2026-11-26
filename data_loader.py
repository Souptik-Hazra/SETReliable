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


def _parse_month(values):
    text = values.astype("string").str.strip()
    return pd.to_datetime(text, format="%Y%m", errors="coerce")


def load_raw_data(path=INPUT_FILE, chunksize=CHUNK_SIZE):
    """Read the 1GB NHSBSA CSV in chunks and normalize its eight-column schema."""
    path = str(path)
    parts = []
    for chunk in pd.read_csv(
        path, dtype={YEAR_MONTH_COL: "string", ODS_CODE_COL: "string",
                     MEDICINE_CODE_COL: "string", UNIT_ID_COL: "string"},
        chunksize=chunksize, low_memory=False,
    ):
        missing = [c for c in REQUIRED_COLUMNS if c not in chunk.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        chunk["DATE"] = _parse_month(chunk[YEAR_MONTH_COL])
        chunk[QUANTITY_COL] = pd.to_numeric(chunk[QUANTITY_COL], errors="coerce")
        chunk[COST_COL] = pd.to_numeric(chunk[COST_COL], errors="coerce")
        chunk["RAW_QUANTITY"] = chunk[QUANTITY_COL]
        chunk["NEGATIVE_QUANTITY_FLAG"] = (chunk[QUANTITY_COL] < 0).astype("int8")
        chunk["NEGATIVE_COST_FLAG"] = (chunk[COST_COL] < 0).astype("int8")
        chunk["DEMAND"] = chunk[QUANTITY_COL].where(chunk[QUANTITY_COL] >= 0)

        chunk = chunk.dropna(subset=[
            "DATE", ODS_CODE_COL, MEDICINE_CODE_COL, UNIT_COL, QUANTITY_COL
        ])
        for col in [ODS_CODE_COL, MEDICINE_CODE_COL, MEDICINE_NAME_COL, UNIT_ID_COL, UNIT_COL]:
            chunk[col] = chunk[col].astype("string").str.strip()

        chunk = chunk[
            chunk[ODS_CODE_COL].ne("") &
            chunk[MEDICINE_CODE_COL].ne("") &
            chunk[UNIT_COL].ne("")
        ].copy()

        chunk["MEDICINE_HOSPITAL_ID"] = (
            chunk[ODS_CODE_COL] + "_" + chunk[MEDICINE_CODE_COL] + "_" + chunk[UNIT_COL]
        )
        parts.append(chunk)

    if not parts:
        raise ValueError(f"No valid records found in {path}")

    df = pd.concat(parts, ignore_index=True)
    keys = ["DATE", ODS_CODE_COL, MEDICINE_CODE_COL, UNIT_COL]
    df = (
        df.groupby(keys, as_index=False, dropna=False)
        .agg(
            **{
                "DEMAND": ("DEMAND", "sum"),
                "RAW_QUANTITY": ("RAW_QUANTITY", "sum"),
                "INDICATIVE_COST_TOTAL": (COST_COL, "sum"),
                "COST_RECORDS": (COST_COL, "count"),
                "NEGATIVE_QUANTITY_COUNT": ("NEGATIVE_QUANTITY_FLAG", "sum"),
                "NEGATIVE_COST_COUNT": ("NEGATIVE_COST_FLAG", "sum"),
                MEDICINE_NAME_COL: (MEDICINE_NAME_COL, "first"),
                UNIT_ID_COL: (UNIT_ID_COL, "first"),
            }
        )
    )
    df["YEAR_MONTH"] = df["DATE"].dt.strftime("%Y%m")
    df["COST_PER_RECORD"] = (
        df["INDICATIVE_COST_TOTAL"] / df["COST_RECORDS"].replace(0, pd.NA)
    )
    df["MEDICINE_HOSPITAL_ID"] = (
        df[ODS_CODE_COL].astype(str) + "_" +
        df[MEDICINE_CODE_COL].astype(str) + "_" +
        df[UNIT_COL].astype(str)
    )
    return df.sort_values(["MEDICINE_HOSPITAL_ID", "DATE"]).reset_index(drop=True)


def filter_series(df, min_months=MIN_HISTORY_MONTHS):
    counts = df.groupby("MEDICINE_HOSPITAL_ID")["DATE"].nunique()
    valid = counts[counts >= min_months].index
    return df[df["MEDICINE_HOSPITAL_ID"].isin(valid)].copy()


def create_continuous_monthly_series(df):
    """Insert absent calendar months as NaN, never as assumed zero demand."""
    frames = []
    for series_id, group in df.groupby("MEDICINE_HOSPITAL_ID", sort=False):
        group = group.sort_values("DATE").set_index("DATE")
        index = pd.date_range(group.index.min(), group.index.max(), freq="MS")
        group = group.reindex(index)
        group.index.name = "DATE"
        group = group.reset_index()
        group["MEDICINE_HOSPITAL_ID"] = series_id
        for col in [ODS_CODE_COL, MEDICINE_CODE_COL, MEDICINE_NAME_COL, UNIT_ID_COL, UNIT_COL]:
            group[col] = group[col].ffill().bfill()
        group["YEAR_MONTH"] = group["DATE"].dt.strftime("%Y%m")
        group["OBSERVATION_MISSING"] = group["DEMAND"].isna().astype("int8")
        frames.append(group)
    if not frames:
        raise ValueError("No eligible medicine-Trust series remain after history filtering.")
    return pd.concat(frames, ignore_index=True)
