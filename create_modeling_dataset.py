
import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

data_folder = r"C:\Users\paula\OneDrive\Desktop\SETPart2\data"

input_file = os.path.join(
    data_folder,
    "combined_secondary_care_medicines.csv"
)

output_file = os.path.join(
    data_folder,
    "modeling_dataset.csv"
)

exclusion_file = os.path.join(
    data_folder,
    "modeling_dataset_unit_exclusions.csv"
)

# --------------------------------------------------
# 2. REQUIRED COLUMNS
# --------------------------------------------------

columns = [
    "YEAR_MONTH",
    "ODS_CODE",
    "VMP_SNOMED_CODE",
    "VMP_PRODUCT_NAME",
    "UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANITY_IN_VMP_UNIT",
    "VMP_UDFS_UNIT_OF_MEASURE_NAME",
    "TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE"
]

chunk_size = 200000

# --------------------------------------------------
# 3. READ DATA IN CHUNKS AND AGGREGATE
# --------------------------------------------------

monthly_parts = []

print("Reading original NHS dataset...")

for chunk in pd.read_csv(
    input_file,
    usecols=columns,
    dtype={
        "YEAR_MONTH": str,
        "ODS_CODE": str,
        "VMP_SNOMED_CODE": str
    },
    chunksize=chunk_size,
    low_memory=False
):

    # Select the applicable unit
    chunk["UNIT"] = chunk["VMP_UDFS_UNIT_OF_MEASURE_NAME"].fillna(
        chunk["UNIT_OF_MEASURE_NAME"]
    )

    # Select the corresponding quantity
    chunk["QUANTITY"] = chunk[
        "TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE"
    ].fillna(
        chunk["TOTAL_QUANITY_IN_VMP_UNIT"]
    )

    # Clean unit names
    chunk["UNIT"] = (
        chunk["UNIT"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Convert quantity to numeric
    chunk["QUANTITY"] = pd.to_numeric(
        chunk["QUANTITY"],
        errors="coerce"
    )

    # Remove unknown units and missing quantities
    chunk = chunk[
        chunk["UNIT"].notna()
        & (chunk["UNIT"] != "")
        & (chunk["UNIT"] != "NO VALUE")
        & (chunk["UNIT"] != "UNKNOWN")
        & chunk["QUANTITY"].notna()
    ]

    # Aggregate within each chunk
    grouped = chunk.groupby(
        [
            "YEAR_MONTH",
            "ODS_CODE",
            "VMP_SNOMED_CODE",
            "VMP_PRODUCT_NAME",
            "UNIT"
        ],
        as_index=False
    ).agg(
        DEMAND=("QUANTITY", lambda x: x.sum(min_count=1))
    )

    monthly_parts.append(grouped)

    print("Chunk processed.")

# Combine chunk-level aggregates
monthly_data = pd.concat(
    monthly_parts,
    ignore_index=True
)

# Aggregate again because groups may appear in multiple chunks
monthly_data = monthly_data.groupby(
    [
        "YEAR_MONTH",
        "ODS_CODE",
        "VMP_SNOMED_CODE",
        "VMP_PRODUCT_NAME",
        "UNIT"
    ],
    as_index=False
).agg(
    DEMAND=("DEMAND", lambda x: x.sum(min_count=1))
)

print("\nAggregation completed.")

# --------------------------------------------------
# 4. IDENTIFY UNIT-INCONSISTENT SERIES
# --------------------------------------------------

unit_counts = monthly_data.groupby(
    ["ODS_CODE", "VMP_SNOMED_CODE"]
)["UNIT"].nunique()

stable_series = unit_counts[unit_counts == 1].index
unstable_series = unit_counts[unit_counts > 1].index

print("Stable medicine-Trust series:", len(stable_series))
print("Series with multiple units:", len(unstable_series))

# Save the excluded series for review
excluded = monthly_data[
    monthly_data.set_index(
        ["ODS_CODE", "VMP_SNOMED_CODE"]
    ).index.isin(unstable_series)
].copy()

excluded.to_csv(exclusion_file, index=False)

# Keep only series with one consistent unit
monthly_data = monthly_data[
    monthly_data.set_index(
        ["ODS_CODE", "VMP_SNOMED_CODE"]
    ).index.isin(stable_series)
].copy()

print("Stable-unit data selected.")

# --------------------------------------------------
# 5. PREPARE MONTHLY TIME SERIES
# --------------------------------------------------

monthly_data["YEAR_MONTH"] = pd.to_datetime(
    monthly_data["YEAR_MONTH"],
    format="%Y%m"
)

monthly_data = monthly_data.sort_values(
    ["ODS_CODE", "VMP_SNOMED_CODE", "YEAR_MONTH"]
)

# Create features separately for each medicine and Trust
feature_data = []

for (trust, medicine), group in monthly_data.groupby(
    ["ODS_CODE", "VMP_SNOMED_CODE"]
):

    group = group.sort_values("YEAR_MONTH").copy()

    # Create continuous monthly dates only within the observed period
    full_dates = pd.date_range(
        start=group["YEAR_MONTH"].min(),
        end=group["YEAR_MONTH"].max(),
        freq="MS"
    )

    group = group.set_index("YEAR_MONTH").reindex(full_dates)

    group.index.name = "YEAR_MONTH"

    # Restore identifiers
    group["ODS_CODE"] = trust
    group["VMP_SNOMED_CODE"] = medicine

    # Restore product and unit names
    group["VMP_PRODUCT_NAME"] = group[
        "VMP_PRODUCT_NAME"
    ].ffill().bfill()

    group["UNIT"] = group["UNIT"].ffill().bfill()

    # Missing months within the observed time range
    # are treated as zero recorded demand for this first version
    group["DEMAND"] = group["DEMAND"].fillna(0)

    # Calendar features
    group["YEAR"] = group.index.year
    group["MONTH"] = group.index.month

    # Lag features
    group["LAG_1"] = group["DEMAND"].shift(1)
    group["LAG_2"] = group["DEMAND"].shift(2)
    group["LAG_3"] = group["DEMAND"].shift(3)
    group["LAG_6"] = group["DEMAND"].shift(6)
    group["LAG_12"] = group["DEMAND"].shift(12)

    # Rolling features using only previous months
    previous_demand = group["DEMAND"].shift(1)

    group["ROLLING_MEAN_3"] = previous_demand.rolling(
        window=3, min_periods=3
    ).mean()

    group["ROLLING_STD_3"] = previous_demand.rolling(
        window=3, min_periods=3
    ).std()

    group["ROLLING_MEAN_6"] = previous_demand.rolling(
        window=6, min_periods=6
    ).mean()

    group["ROLLING_STD_6"] = previous_demand.rolling(
        window=6, min_periods=6
    ).std()

    group = group.reset_index()

    feature_data.append(group)

# Combine all series
modeling_data = pd.concat(
    feature_data,
    ignore_index=True
)

# --------------------------------------------------
# 6. SAVE MODELLING DATASET
# --------------------------------------------------

modeling_data = modeling_data.sort_values(
    ["ODS_CODE", "VMP_SNOMED_CODE", "YEAR_MONTH"]
)

modeling_data.to_csv(
    output_file,
    index=False
)

# --------------------------------------------------
# 7. FINAL SUMMARY
# --------------------------------------------------

print("\n======================================")
print("MODELLING DATASET CREATED")
print("======================================")

print("Total rows:", len(modeling_data))
print("Unique medicines:", modeling_data["VMP_SNOMED_CODE"].nunique())
print("Unique Trusts:", modeling_data["ODS_CODE"].nunique())
print("Unique medicine-Trust series:",
      modeling_data.groupby(["ODS_CODE", "VMP_SNOMED_CODE"]).ngroups)

print("\nMissing feature values:")
print(modeling_data[
    [
        "LAG_1", "LAG_2", "LAG_3",
        "LAG_6", "LAG_12",
        "ROLLING_MEAN_3", "ROLLING_STD_3",
        "ROLLING_MEAN_6", "ROLLING_STD_6"
    ]
].isna().sum())

print("\nSaved files:")
print(output_file)
print(exclusion_file)
