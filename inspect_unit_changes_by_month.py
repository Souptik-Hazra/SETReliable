import pandas as pd
from pathlib import Path

data_folder = Path(
    r"C:\Users\paula\OneDrive\Desktop\SETPart2\data"
)

input_file = data_folder / "combined_secondary_care_medicines.csv"
output_folder = data_folder / "data_quality_reports"

medicine_codes = [
    "34938511000001103",
    "21929811000001100",
    "19697811000001108",
    "10161711000001105"
]

columns = [
    "YEAR_MONTH",
    "VMP_SNOMED_CODE",
    "VMP_PRODUCT_NAME",
    "UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANITY_IN_VMP_UNIT",
    "VMP_UDFS_UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE",
    "VMP_UNIT_DOSE_UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANTITY_IN_VMP_UNIT_DOSE_UNIT_OF_MEASURE"
]

results = []

for chunk in pd.read_csv(
    input_file,
    usecols=columns,
    chunksize=200_000,
    low_memory=False
):

    chunk["VMP_SNOMED_CODE"] = (
        chunk["VMP_SNOMED_CODE"].astype(str).str.strip()
    )

    filtered = chunk[
        chunk["VMP_SNOMED_CODE"].isin(medicine_codes)
    ].copy()

    if not filtered.empty:
        results.append(filtered)

    print("Chunk processed")

# Combine extracted records only
df = pd.concat(results, ignore_index=True)

# Save detailed records
df.to_csv(
    output_folder / "four_medicines_unit_investigation.csv",
    index=False
)

# -----------------------------------
# Monthly summary by unit combination
# -----------------------------------

df["YEAR_MONTH"] = df["YEAR_MONTH"].astype(str)

summary = df.groupby(
    [
        "YEAR_MONTH",
        "VMP_SNOMED_CODE",
        "VMP_PRODUCT_NAME",
        "UNIT_OF_MEASURE_NAME",
        "VMP_UDFS_UNIT_OF_MEASURE_NAME",
        "VMP_UNIT_DOSE_UNIT_OF_MEASURE_NAME"
    ],
    dropna=False
).agg(
    RECORD_COUNT=("VMP_SNOMED_CODE", "size"),
    OLD_QUANTITY_TOTAL=("TOTAL_QUANITY_IN_VMP_UNIT", "sum"),
    UDFS_QUANTITY_TOTAL=(
        "TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE", "sum"
    ),
    UNIT_DOSE_QUANTITY_TOTAL=(
        "TOTAL_QUANTITY_IN_VMP_UNIT_DOSE_UNIT_OF_MEASURE", "sum"
    )
).reset_index()

summary = summary.sort_values(
    ["VMP_SNOMED_CODE", "YEAR_MONTH"]
)

summary.to_csv(
    output_folder / "four_medicines_monthly_unit_summary.csv",
    index=False
)

print("\n===== MONTHLY UNIT INVESTIGATION =====")
print(summary.to_string(index=False))

print("\nReports saved successfully.")