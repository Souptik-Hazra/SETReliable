"""Disk-backed SCMD data-quality audit using DuckDB global aggregations."""
from pathlib import Path
import duckdb
import pandas as pd

from config import INPUT_FILE
from data_loader import REQUIRED_COLUMNS


def audit(path=INPUT_FILE, chunksize=None, memory_limit="4GB", temp_directory=None):
    """Audit exact duplicate rows and duplicate series-month keys out of core.

    DuckDB groups directly over the CSV and spills intermediate state to disk.
    No Python loop or one-row-at-a-time database insert is used.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(pd.read_csv(path, nrows=0).columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    source = str(path).replace("'", "''")
    conn = duckdb.connect(database=":memory:")
    try:
        conn.execute(f"SET memory_limit = '{memory_limit}'")
        if temp_directory:
            directory = str(Path(temp_directory).resolve()).replace("'", "''")
            conn.execute(f"SET temp_directory = '{directory}'")
        conn.execute(f"""
            CREATE TEMP VIEW scmd_source AS
            SELECT * FROM read_csv_auto('{source}', all_varchar=true, header=true)
        """)
        numeric = conn.execute("""
            SELECT
                count(*) AS rows,
                count(*) FILTER (WHERE try_cast(INDICATIVE_COST AS DOUBLE) IS NULL) AS missing_cost,
                count(*) FILTER (WHERE try_cast(TOTAL_QUANITY_IN_VMP_UNIT AS DOUBLE) < 0) AS negative_quantity,
                count(*) FILTER (WHERE try_cast(TOTAL_QUANITY_IN_VMP_UNIT AS DOUBLE) = 0) AS zero_quantity,
                count(*) FILTER (WHERE try_cast(INDICATIVE_COST AS DOUBLE) < 0) AS negative_cost,
                count(*) FILTER (WHERE try_cast(TOTAL_QUANITY_IN_VMP_UNIT AS DOUBLE) IS NULL) AS invalid_quantity,
                count(*) FILTER (WHERE try_strptime(trim(YEAR_MONTH), '%Y%m') IS NULL) AS invalid_month
            FROM scmd_source
        """).df().iloc[0].to_dict()
        duplicates = conn.execute("""
            SELECT
                coalesce(sum(row_count - 1), 0) AS exact_duplicate_rows
            FROM (
                SELECT count(*) AS row_count
                FROM scmd_source
                GROUP BY ALL
                HAVING count(*) > 1
            )
        """).df().iloc[0].to_dict()
        series_duplicates = conn.execute("""
            SELECT coalesce(sum(row_count - 1), 0) AS duplicate_series_month_keys
            FROM (
                SELECT count(*) AS row_count
                FROM (
                    SELECT
                        trim(YEAR_MONTH) AS YEAR_MONTH,
                        nullif(trim(ODS_CODE), '') AS ODS_CODE,
                        nullif(trim(VMP_SNOMED_CODE), '') AS VMP_SNOMED_CODE,
                        coalesce(nullif(trim(UNIT_OF_MEASURE_IDENTIFIER), ''),
                                 'UNKNOWN_UNIT_ID') AS UNIT_ID
                    FROM scmd_source
                    WHERE try_strptime(trim(YEAR_MONTH), '%Y%m') IS NOT NULL
                      AND nullif(trim(ODS_CODE), '') IS NOT NULL
                      AND nullif(trim(VMP_SNOMED_CODE), '') IS NOT NULL
                )
                GROUP BY YEAR_MONTH, ODS_CODE, VMP_SNOMED_CODE, UNIT_ID
                HAVING count(*) > 1
            )
        """).df().iloc[0].to_dict()
        return {
            **{key: int(value) for key, value in numeric.items()},
            **{key: int(value) for key, value in duplicates.items()},
            **{key: int(value) for key, value in series_duplicates.items()},
        }
    finally:
        conn.close()


if __name__ == "__main__":
    print(audit())
