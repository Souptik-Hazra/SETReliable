import pandas as pd
from collections import defaultdict
from pathlib import Path

# Paths
data_folder = Path(r"C:\Users\paula\OneDrive\Desktop\SETPart2\data")

input_file = data_folder / "combined_secondary_care_medicines.csv"
output_folder = data_folder / "data_quality_reports"
output_folder.mkdir(exist_ok=True)

# Columns needed for this analysis
columns = [
    "YEAR_MONTH",
    "VMP_SNOMED_CODE",
    "VMP_PRODUCT_NAME",
    "UNIT_OF_MEASURE_NAME",
    "VMP_UDFS_UNIT_OF_MEASURE_NAME"
]

# Store units and months for each medicine
medicine_units = defaultdict(set)
medicine_months = defaultdict(set)
medicine_names = {}

# Store monthly unit distribution
monthly_unit_counts = defaultdict(int)

# Treat blank values and explicit "no value" labels as unknown
unknown_units = {"", "NAN", "NONE", "NO VALUE", "UNKNOWN"}

# Read the large file in chunks
for chunk in pd.read_csv(
    input_file,
    usecols=columns,
    chunksize=200_000,
    low_memory=False
):

    # Combine old and new UDFS unit columns
    old_units = chunk["UNIT_OF_MEASURE_NAME"].fillna("").astype(str).str.strip()
    new_units = chunk["VMP_UDFS_UNIT_OF_MEASURE_NAME"].fillna("").astype(str).str.strip()

    chunk["UDFS_UNIT"] = new_units.where(new_units != "", old_units)

    chunk["UDFS_UNIT"] = chunk["UDFS_UNIT"].str.upper()
    chunk["VMP_SNOMED_CODE"] = chunk["VMP_SNOMED_CODE"].astype(str)
    chunk["YEAR_MONTH"] = chunk["YEAR_MONTH"].astype(str)

    # Remove missing medicine codes
    chunk = chunk[
        chunk["VMP_SNOMED_CODE"].notna()
        & (chunk["VMP_SNOMED_CODE"] != "")
    ]

    # Store medicine names
    for code, name in zip(
        chunk["VMP_SNOMED_CODE"],
        chunk["VMP_PRODUCT_NAME"]
    ):
        medicine_names[code] = name

    # Collect units and months per medicine
    for code, unit in zip(
        chunk["VMP_SNOMED_CODE"],
        chunk["UDFS_UNIT"]
    ):
        if unit not in unknown_units:
            medicine_units[code].add(unit)

    for code, month in zip(
        chunk["VMP_SNOMED_CODE"],
        chunk["YEAR_MONTH"]
    ):
        medicine_months[code].add(month)

    # Count monthly occurrences of each UDFS unit
    valid = chunk[
        ~chunk["UDFS_UNIT"].isin(unknown_units)
    ]

    counts = valid.groupby(
        ["YEAR_MONTH", "UDFS_UNIT"]
    ).size()

    for (month, unit), count in counts.items():
        monthly_unit_counts[(month, unit)] += count

    print("Processed chunk")

print("\nAnalysis completed. Preparing reports...")

# ---------------------------------
# 1. Medicine-level unit consistency
# ---------------------------------

summary_rows = []

for code, units in medicine_units.items():

    months = medicine_months[code]

    if len(units) == 1:
        status = "Consistent"
    elif len(units) > 1:
        status = "Multiple units"
    else:
        status = "Unknown"

    summary_rows.append({
        "VMP_SNOMED_CODE": code,
        "VMP_PRODUCT_NAME": medicine_names.get(code, ""),
        "MONTHS_PRESENT": len(months),
        "NUMBER_OF_UNITS": len(units),
        "UNITS_FOUND": " | ".join(sorted(units)),
        "UNIT_STATUS": status
    })

unit_summary = pd.DataFrame(summary_rows)

unit_summary.to_csv(
    output_folder / "medicine_unit_consistency.csv",
    index=False
)

# ---------------------------------
# 2. Monthly unit distribution
# ---------------------------------

monthly_rows = []

for (month, unit), count in monthly_unit_counts.items():
    monthly_rows.append({
        "YEAR_MONTH": month,
        "UDFS_UNIT": unit,
        "RECORD_COUNT": count
    })

monthly_summary = pd.DataFrame(monthly_rows)

monthly_summary = monthly_summary.sort_values(
    ["YEAR_MONTH", "UDFS_UNIT"]
)

monthly_summary.to_csv(
    output_folder / "monthly_udfs_unit_distribution.csv",
    index=False
)

# ---------------------------------
# 3. Overall summary
# ---------------------------------

print("\n===== UNIT CONSISTENCY SUMMARY =====")

print("Total medicines analyzed:", len(unit_summary))

print("\nUnit consistency categories:")
print(unit_summary["UNIT_STATUS"].value_counts())

print("\nResults saved inside:")
print(output_folder)

print("\nCompleted successfully.")