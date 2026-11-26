"""Disk-backed audit of unit-name variation within Trust/medicine/unit identity."""
from pathlib import Path

import duckdb
import pandas as pd

from config import INPUT_FILE


REQUIRED_COLUMNS = {
    "ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER",
    "UNIT_OF_MEASURE_NAME", "VMP_PRODUCT_NAME",
}


def analyze(path=INPUT_FILE, chunksize=None, memory_limit="4GB", temp_directory=None):
    """Aggregate unit-name variants in DuckDB instead of iterating over 10m rows."""
    missing = REQUIRED_COLUMNS - set(pd.read_csv(path, nrows=0).columns)
    if missing:
        raise ValueError(f"Missing current-schema columns: {sorted(missing)}")

    source = str(path).replace("'", "''")
    conn = duckdb.connect(database=":memory:")
    try:
        conn.execute(f"SET memory_limit = '{memory_limit}'")
        if temp_directory:
            directory = str(Path(temp_directory).resolve()).replace("'", "''")
            conn.execute(f"SET temp_directory = '{directory}'")
        result = conn.execute(f"""
            SELECT
                nullif(trim(ODS_CODE), '') AS ODS_CODE,
                nullif(trim(VMP_SNOMED_CODE), '') AS VMP_SNOMED_CODE,
                coalesce(nullif(trim(UNIT_OF_MEASURE_IDENTIFIER), ''),
                         'UNKNOWN_UNIT_ID') AS UNIT_OF_MEASURE_IDENTIFIER,
                first(nullif(trim(VMP_PRODUCT_NAME), '')) AS VMP_PRODUCT_NAME,
                list_sort(list_distinct(list(nullif(trim(UNIT_OF_MEASURE_NAME), ''))
                    FILTER (WHERE nullif(trim(UNIT_OF_MEASURE_NAME), '') IS NOT NULL)))
                    AS UNIT_NAMES
            FROM read_csv_auto('{source}', all_varchar=true, header=true)
            WHERE nullif(trim(ODS_CODE), '') IS NOT NULL
              AND nullif(trim(VMP_SNOMED_CODE), '') IS NOT NULL
            GROUP BY ODS_CODE, VMP_SNOMED_CODE, UNIT_OF_MEASURE_IDENTIFIER
        """).df()

        result["UNIT_NAME_VARIANTS"] = result["UNIT_NAMES"].map(
            lambda values: len(values) if values else 0
        ).astype("int16")
        result["UNIT_NAMES"] = result["UNIT_NAMES"].map(
            lambda values: " | ".join(values) if values else ""
        )
        return result.sort_values(
            ["UNIT_NAME_VARIANTS", "ODS_CODE", "VMP_SNOMED_CODE"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    finally:
        conn.close()


if __name__ == "__main__":
    result = analyze()
    print(result.head(100).to_string(index=False))
    print(f"\nSeries with multiple unit names: {(result['UNIT_NAME_VARIANTS'] > 1).sum():,}")
