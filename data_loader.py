import pandas as pd
import duckdb

from config import (
    INPUT_FILE, YEAR_MONTH_COL, ODS_CODE_COL, MEDICINE_CODE_COL,
    MEDICINE_NAME_COL, UNIT_ID_COL, UNIT_COL, QUANTITY_COL, COST_COL,
    MIN_HISTORY_MONTHS,
)

REQUIRED_COLUMNS = [
    YEAR_MONTH_COL, ODS_CODE_COL, MEDICINE_CODE_COL, MEDICINE_NAME_COL,
    UNIT_ID_COL, UNIT_COL, QUANTITY_COL, COST_COL,
]
SERIES_KEYS = ["DATE", ODS_CODE_COL, MEDICINE_CODE_COL, UNIT_ID_COL]


def _clean_text(series):
    return series.astype("string").str.strip().replace("", pd.NA)


def load_raw_data(path=INPUT_FILE, chunksize=None, memory_limit="4GB"):
    """Out-of-core aggregation for the current pre-June-2026 SCMD CSV.

    DuckDB performs the global group-by on disk and can spill intermediate
    state to its temporary directory. Only the aggregated monthly result is
    materialised as a pandas DataFrame.
    """
    source = str(path).replace("'", "''")
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(REQUIRED_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    conn = duckdb.connect(database=":memory:")
    try:
        conn.execute(f"SET memory_limit = '{memory_limit}'")
        conn.execute("""
            CREATE TEMP VIEW scmd_source AS
            SELECT
                trim(YEAR_MONTH) AS YEAR_MONTH,
                nullif(trim(ODS_CODE), '') AS ODS_CODE,
                nullif(trim(VMP_SNOMED_CODE), '') AS VMP_SNOMED_CODE,
                nullif(trim(VMP_PRODUCT_NAME), '') AS VMP_PRODUCT_NAME,
                coalesce(nullif(trim(UNIT_OF_MEASURE_IDENTIFIER), ''), 'UNKNOWN_UNIT_ID') AS UNIT_OF_MEASURE_IDENTIFIER,
                nullif(trim(UNIT_OF_MEASURE_NAME), '') AS UNIT_OF_MEASURE_NAME,
                try_cast(trim(TOTAL_QUANITY_IN_VMP_UNIT) AS DOUBLE) AS QUANTITY,
                try_cast(trim(INDICATIVE_COST) AS DOUBLE) AS COST,
                try_strptime(trim(YEAR_MONTH), '%Y%m')::DATE AS DATE
            FROM read_csv_auto('{source}', all_varchar=true, header=true)
        """)

        q = conn.execute("""
            SELECT
                count(*) AS source_rows,
                count(*) FILTER (WHERE DATE IS NULL) AS invalid_month_rows,
                count(*) FILTER (WHERE ODS_CODE IS NULL OR VMP_SNOMED_CODE IS NULL) AS missing_identity_rows,
                count(*) FILTER (WHERE QUANTITY IS NULL) AS invalid_quantity_rows,
                count(*) FILTER (WHERE COST IS NULL) AS missing_cost_rows,
                count(*) FILTER (WHERE QUANTITY < 0) AS negative_quantity_rows,
                count(*) FILTER (WHERE QUANTITY = 0) AS zero_quantity_rows,
                count(*) FILTER (WHERE COST < 0) AS negative_cost_rows
            FROM scmd_source
        """).df().iloc[0].to_dict()
        quality = {key: int(value) for key, value in q.items()}

        data = conn.execute("""
            SELECT
                DATE, ODS_CODE, VMP_SNOMED_CODE AS VMP_SNOMED_CODE,
                UNIT_OF_MEASURE_IDENTIFIER,
                coalesce(sum(QUANTITY) FILTER (WHERE QUANTITY > 0), 0.0) AS GROSS_POSITIVE_QUANTITY,
                coalesce(sum(QUANTITY) FILTER (WHERE QUANTITY < 0), 0.0) AS NEGATIVE_QUANTITY_TOTAL,
                sum(QUANTITY) AS NET_QUANTITY,
                count(*) FILTER (WHERE QUANTITY < 0) AS NEGATIVE_QUANTITY_COUNT,
                count(*) FILTER (WHERE QUANTITY = 0) AS ZERO_QUANTITY_COUNT,
                count(*) FILTER (WHERE COST < 0) AS NEGATIVE_COST_COUNT,
                sum(COST) AS INDICATIVE_COST_TOTAL,
                count(COST) AS COST_RECORDS,
                count(*) AS SOURCE_RECORD_COUNT,
                first(VMP_PRODUCT_NAME) AS VMP_PRODUCT_NAME,
                list_sort(list_distinct(list(UNIT_OF_MEASURE_NAME))) AS UNIT_NAME_SET
            FROM scmd_source
            WHERE DATE IS NOT NULL
              AND ODS_CODE IS NOT NULL
              AND VMP_SNOMED_CODE IS NOT NULL
              AND QUANTITY IS NOT NULL
            GROUP BY DATE, ODS_CODE, VMP_SNOMED_CODE, UNIT_OF_MEASURE_IDENTIFIER
            ORDER BY ODS_CODE, VMP_SNOMED_CODE, UNIT_OF_MEASURE_IDENTIFIER, DATE
        """).df()

        if data.empty:
            raise ValueError(f"No valid records found in {path}")

        data["UNIT_NAME_SET"] = data["UNIT_NAME_SET"].map(
            lambda values: tuple(values) if values else ()
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
        name_counts = data.groupby("MEDICINE_HOSPITAL_ID")["UNIT_NAME_SET"].agg(
            lambda values: len(set().union(*(set(value) for value in values)))
        )
        data["UNIT_NAME_VARIANTS"] = data["MEDICINE_HOSPITAL_ID"].map(name_counts).astype("int16")
        quality["aggregated_monthly_rows"] = len(data)
        quality["unit_name_conflict_series"] = int((name_counts > 1).sum())
        data = data.sort_values(["MEDICINE_HOSPITAL_ID", "DATE"]).reset_index(drop=True)
        data.attrs["quality_report"] = quality
        return data
    finally:
        conn.close()


def filter_series(df, min_months=MIN_HISTORY_MONTHS, target="GROSS_POSITIVE_QUANTITY"):
    if target not in df:
        raise ValueError(f"Unknown series filter target: {target}")
    counts = df.loc[df[target].notna()].groupby("MEDICINE_HOSPITAL_ID")[target].count()
    valid_ids = counts[counts >= min_months].index
    return df[df["MEDICINE_HOSPITAL_ID"].isin(valid_ids)].copy()


def create_continuous_monthly_series(df):
    """Reindex calendar months; source-observed zeroes differ from absent months."""
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
