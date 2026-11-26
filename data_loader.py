import pandas as pd

from config import (
    INPUT_FILE,
    YEAR_MONTH_COL,
    ODS_CODE_COL,
    MEDICINE_CODE_COL,
    MEDICINE_NAME_COL,
    QUANTITY_COL,
    UNIT_COL,
    COST_COL,
    MIN_HISTORY_MONTHS,
)


def load_raw_data():
    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        YEAR_MONTH_COL,
        ODS_CODE_COL,
        MEDICINE_CODE_COL,
        MEDICINE_NAME_COL,
        QUANTITY_COL,
        UNIT_COL,
        COST_COL,
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df[QUANTITY_COL] = pd.to_numeric(df[QUANTITY_COL], errors="coerce")
    df[COST_COL] = pd.to_numeric(df[COST_COL], errors="coerce")

    df = df.dropna(
        subset=[
            YEAR_MONTH_COL,
            ODS_CODE_COL,
            MEDICINE_CODE_COL,
            QUANTITY_COL,
        ]
    )

    df["DATE"] = pd.to_datetime(
        df[YEAR_MONTH_COL].astype(str),
        errors="coerce",
        format="%Y-%m",
    )

    if df["DATE"].isna().any():
        df["DATE"] = pd.to_datetime(
            df[YEAR_MONTH_COL].astype(str),
            errors="coerce",
            format="%Y/%m",
        )

    df = df.dropna(subset=["DATE"])

    df["DEMAND"] = df[QUANTITY_COL].clip(lower=0)

    df["MEDICINE_HOSPITAL_ID"] = (
        df[ODS_CODE_COL].astype(str)
        + "_"
        + df[MEDICINE_CODE_COL].astype(str)
    )

    df = df.sort_values(
        by=[
            "MEDICINE_HOSPITAL_ID",
            "DATE",
        ]
    )

    return df


def filter_series(df):
    series_counts = (
        df.groupby("MEDICINE_HOSPITAL_ID")["DATE"]
        .nunique()
        .reset_index(name="N_MONTHS")
    )

    valid_ids = series_counts[
        series_counts["N_MONTHS"] >= MIN_HISTORY_MONTHS
    ]["MEDICINE_HOSPITAL_ID"]

    df = df[df["MEDICINE_HOSPITAL_ID"].isin(valid_ids)].copy()

    return df


def create_continuous_monthly_series(df):
    output_frames = []

    for series_id, group in df.groupby("MEDICINE_HOSPITAL_ID"):
        group = group.sort_values("DATE").copy()

        full_index = pd.date_range(
            start=group["DATE"].min(),
            end=group["DATE"].max(),
            freq="MS",
        )

        group = group.set_index("DATE")

        group = group.reindex(full_index)

        group.index.name = "DATE"
        group = group.reset_index()

        group["MEDICINE_HOSPITAL_ID"] = series_id

        first_row = group.iloc[0]

        group[ODS_CODE_COL] = first_row[ODS_CODE_COL]
        group[MEDICINE_CODE_COL] = first_row[MEDICINE_CODE_COL]
        group[MEDICINE_NAME_COL] = first_row[MEDICINE_NAME_COL]
        group[UNIT_COL] = first_row[UNIT_COL]
        group[COST_COL] = first_row[COST_COL]

        group["DEMAND"] = group["DEMAND"].fillna(0)

        output_frames.append(group)

    df = pd.concat(output_frames, ignore_index=True)

    return df