
import pandas as pd
import os

# File paths
data_folder = r"C:\Users\paula\OneDrive\Desktop\SETPart2\data"
input_file = os.path.join(data_folder, "combined_secondary_care_medicines.csv")

output_folder = os.path.join(data_folder, "data_quality_reports")
os.makedirs(output_folder, exist_ok=True)

# The 4 medicines being investigated
medicine_codes = [
    "34938511000001103",  # MMR vaccine
    "21929811000001100",  # Bevacizumab pre-filled syringe
    "19697811000001108",   # Bevacizumab vial
    "10161711000001105"    # Levodopa/Carbidopa intestinal gel
]

# Columns required for this investigation
columns = [
    "YEAR_MONTH",
    "ODS_CODE",
    "VMP_SNOMED_CODE",
    "VMP_PRODUCT_NAME",
    "UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANITY_IN_VMP_UNIT",
    "VMP_UDFS_UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE",
    "VMP_UNIT_DOSE_UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANTITY_IN_VMP_UNIT_DOSE_UNIT_OF_MEASURE"
]

# Read the dataset in chunks
chunk_size = 200000

filtered_data = []

print("Investigating the four medicines at Trust level...")

for chunk in pd.read_csv(
    input_file,
    usecols=columns,
    dtype={"VMP_SNOMED_CODE": str, "ODS_CODE": str},
    chunksize=chunk_size,
    low_memory=False
):

    # Keep only the four selected medicines
    chunk = chunk[
        chunk["VMP_SNOMED_CODE"].isin(medicine_codes)
    ]

    if not chunk.empty:
        filtered_data.append(chunk)

# Combine only the filtered records
medicine_data = pd.concat(filtered_data, ignore_index=True)

# Sort the data for easier investigation
medicine_data = medicine_data.sort_values(
    by=["VMP_SNOMED_CODE", "ODS_CODE", "YEAR_MONTH"]
)

# Preserve missing quantities as missing (NaN)
# Do not replace them with zero
quantity_columns = [
    "TOTAL_QUANITY_IN_VMP_UNIT",
    "TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE",
    "TOTAL_QUANTITY_IN_VMP_UNIT_DOSE_UNIT_OF_MEASURE"
]

for col in quantity_columns:
    medicine_data[col] = pd.to_numeric(
        medicine_data[col], errors="coerce"
    )

# Calculate ratio only when both quantities are available and positive
medicine_data["UDFS_TO_UNIT_DOSE_RATIO"] = (
    medicine_data["TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE"]
    / medicine_data["TOTAL_QUANTITY_IN_VMP_UNIT_DOSE_UNIT_OF_MEASURE"]
)

medicine_data.loc[
    (medicine_data["TOTAL_QUANTITY_IN_VMP_UNIT_DOSE_UNIT_OF_MEASURE"] <= 0)
    | (medicine_data["TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE"] <= 0),
    "UDFS_TO_UNIT_DOSE_RATIO"
] = pd.NA

# Save filtered records
detail_file = os.path.join(
    output_folder,
    "four_medicines_trust_level_detail.csv"
)

medicine_data.to_csv(detail_file, index=False)

# Create a unit-level summary
summary = medicine_data.groupby(
    [
        "VMP_SNOMED_CODE",
        "VMP_PRODUCT_NAME",
        "ODS_CODE",
        "YEAR_MONTH"
    ],
    dropna=False
).agg(
    OLD_UNITS=("UNIT_OF_MEASURE_NAME", "first"),
    UDFS_UNITS=("VMP_UDFS_UNIT_OF_MEASURE_NAME", "first"),
    UNIT_DOSE_UNITS=("VMP_UNIT_DOSE_UNIT_OF_MEASURE_NAME", "first"),
    RECORD_COUNT=("YEAR_MONTH", "size"),
    OLD_QUANTITY=("TOTAL_QUANITY_IN_VMP_UNIT", "sum"),
    UDFS_QUANTITY=("TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE", "sum"),
    UNIT_DOSE_QUANTITY=("TOTAL_QUANTITY_IN_VMP_UNIT_DOSE_UNIT_OF_MEASURE", "sum"),
    AVG_UDFS_TO_UNIT_DOSE_RATIO=("UDFS_TO_UNIT_DOSE_RATIO", "mean")
).reset_index()

# Save the summary
summary_file = os.path.join(
    output_folder,
    "four_medicines_trust_level_summary.csv"
)

summary.to_csv(summary_file, index=False)

print("\nInvestigation completed successfully!")

print("\nFiles saved:")
print(detail_file)
print(summary_file)

print("\nTotal filtered records:", len(medicine_data))
print("Number of Trusts:", medicine_data["ODS_CODE"].nunique())
print("Number of medicine codes:", medicine_data["VMP_SNOMED_CODE"].nunique())
