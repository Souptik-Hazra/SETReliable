
import pandas as pd
import os

# -------------------------------
# File paths
# -------------------------------

folder_path = r"C:\Users\paula\OneDrive\Desktop\SETPart2"

input_file = os.path.join(
    folder_path,
    "data",
    "combined_secondary_care_medicines.csv"
)

report_folder = os.path.join(folder_path, "data_quality_reports")

os.makedirs(report_folder, exist_ok=True)

chunk_size = 100000
sample_limit = 5000

quantity_columns = [
    "TOTAL_QUANITY_IN_VMP_UNIT",
    "TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE",
    "TOTAL_QUANTITY_IN_VMP_UNIT_DOSE_UNIT_OF_MEASURE"
]

columns_to_inspect = [
    "YEAR_MONTH",
    "ODS_CODE",
    "VMP_SNOMED_CODE",
    "VMP_PRODUCT_NAME",
    "UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANITY_IN_VMP_UNIT",
    "VMP_UDFS_UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE",
    "VMP_UNIT_DOSE_UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANTITY_IN_VMP_UNIT_DOSE_UNIT_OF_MEASURE",
    "INDICATIVE_COST"
]

# -------------------------------
# Initialize
# -------------------------------

negative_counts = {}
negative_samples = []

# -------------------------------
# Process in chunks
# -------------------------------

for chunk_number, chunk in enumerate(
    pd.read_csv(
        input_file,
        chunksize=chunk_size,
        dtype="string"
    ),
    start=1
):

    print(f"Processing chunk {chunk_number}...")

    for column in quantity_columns:

        values = pd.to_numeric(chunk[column], errors="coerce")

        negative_mask = values < 0

        negative_rows = chunk.loc[negative_mask].copy()

        # Count negative values by month
        if not negative_rows.empty:

            month_counts = negative_rows["YEAR_MONTH"].value_counts()

            for month, count in month_counts.items():

                key = (column, month)

                negative_counts[key] = (
                    negative_counts.get(key, 0) + int(count)
                )

            # Collect samples up to limit per quantity field
            existing_count = sum(
                1 for sample in negative_samples
                if sample["QUANTITY_FIELD"] == column
            )

            remaining = sample_limit - existing_count

            if remaining > 0:

                sample_chunk = negative_rows.head(remaining)

                for _, row in sample_chunk.iterrows():

                    sample_data = {
                        col: row[col]
                        for col in columns_to_inspect
                        if col in row.index
                    }

                    sample_data["QUANTITY_FIELD"] = column

                    sample_data["NEGATIVE_VALUE"] = row[column]

                    negative_samples.append(sample_data)

# -------------------------------
# Save monthly negative counts
# -------------------------------

monthly_report = pd.DataFrame(
    [
        {
            "QUANTITY_FIELD": key[0],
            "YEAR_MONTH": key[1],
            "NEGATIVE_COUNT": count
        }
        for key, count in negative_counts.items()
    ]
)

monthly_report = monthly_report.sort_values(
    ["QUANTITY_FIELD", "YEAR_MONTH"]
)

monthly_report.to_csv(
    os.path.join(report_folder, "negative_quantities_by_month.csv"),
    index=False
)

# -------------------------------
# Save negative sample records
# -------------------------------

samples_df = pd.DataFrame(negative_samples)

samples_df.to_csv(
    os.path.join(report_folder, "negative_quantity_samples.csv"),
    index=False
)

# -------------------------------
# Final output
# -------------------------------

print("\nNegative quantity investigation completed.")

print("\nNegative values by field:")

if not monthly_report.empty:
    print(
        monthly_report.groupby("QUANTITY_FIELD")["NEGATIVE_COUNT"].sum()
    )

print("\nReports saved to:")
print(report_folder)
