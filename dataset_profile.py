"""Profile the configured NHSBSA CSV without writing report files."""
import pandas as pd
from config import INPUT_FILE, CHUNK_SIZE
from data_loader import REQUIRED_COLUMNS


def main():
    total_rows = 0
    missing_cost = 0
    negative_quantity = 0
    negative_cost = 0
    months = set()
    trusts = set()
    medicines = set()
    units = set()

    for chunk in pd.read_csv(
        INPUT_FILE,
        dtype={"YEAR_MONTH": "string", "ODS_CODE": "string",
               "VMP_SNOMED_CODE": "string", "UNIT_OF_MEASURE_IDENTIFIER": "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        missing = set(REQUIRED_COLUMNS) - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        total_rows += len(chunk)
        quantity = pd.to_numeric(chunk["TOTAL_QUANITY_IN_VMP_UNIT"], errors="coerce")
        cost = pd.to_numeric(chunk["INDICATIVE_COST"], errors="coerce")
        missing_cost += int(cost.isna().sum())
        negative_quantity += int((quantity < 0).sum())
        negative_cost += int((cost < 0).sum())
        months.update(chunk["YEAR_MONTH"].dropna().astype(str).unique())
        trusts.update(chunk["ODS_CODE"].dropna().astype(str).unique())
        medicines.update(chunk["VMP_SNOMED_CODE"].dropna().astype(str).unique())
        units.update(chunk["UNIT_OF_MEASURE_IDENTIFIER"].dropna().astype(str).unique())

    print(f"Rows: {total_rows:,}")
    print(f"Months: {len(months):,}")
    print(f"Trusts: {len(trusts):,}")
    print(f"Medicines: {len(medicines):,}")
    print(f"Unit identifiers: {len(units):,}")
    print(f"Missing indicative cost: {missing_cost:,}")
    print(f"Negative quantities: {negative_quantity:,}")
    print(f"Negative indicative costs: {negative_cost:,}")


if __name__ == "__main__":
    main()
