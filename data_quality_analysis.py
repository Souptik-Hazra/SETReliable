
import pandas as pd
import os
import sqlite3

# -------------------------------
# 1. File paths
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

# -------------------------------
# 2. Initialize counters
# -------------------------------

total_rows = 0
monthly_counts = {}
missing_counts = {}
quantity_stats = {}

quantity_columns = [
    "TOTAL_QUANITY_IN_VMP_UNIT",
    "TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE",
    "TOTAL_QUANTITY_IN_VMP_UNIT_DOSE_UNIT_OF_MEASURE"
]

identifier_columns = [
    "YEAR_MONTH",
    "ODS_CODE",
    "VMP_SNOMED_CODE"
]

# -------------------------------
# 3. SQLite duplicate database
# -------------------------------

db_path = os.path.join(report_folder, "audit_temp.db")

if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Hashes stored as TEXT to avoid integer overflow
cursor.execute("""
CREATE TABLE full_rows (
    row_hash TEXT PRIMARY KEY
)
""")

cursor.execute("""
CREATE TABLE medicine_month_keys (
    key_hash TEXT PRIMARY KEY
)
""")

conn.commit()

exact_duplicate_count = 0
duplicate_key_count = 0

# -------------------------------
# 4. Process dataset in chunks
# -------------------------------

for chunk_number, chunk in enumerate(
    pd.read_csv(input_file, chunksize=chunk_size, dtype="string"),
    start=1
):

    print(f"Processing chunk {chunk_number}...")

    total_rows += len(chunk)

    # ---------------------------
    # Monthly record counts
    # ---------------------------

    month_counts = chunk["YEAR_MONTH"].value_counts()

    for month, count in month_counts.items():
        monthly_counts[month] = (
            monthly_counts.get(month, 0) + int(count)
        )

    # ---------------------------
    # Missing value counts
    # ---------------------------

    missing = chunk.isna().sum()

    for column, count in missing.items():
        missing_counts[column] = (
            missing_counts.get(column, 0) + int(count)
        )

    # ---------------------------
    # Quantity analysis
    # ---------------------------

    for column in quantity_columns:

        if column not in chunk.columns:
            continue

        values = pd.to_numeric(chunk[column], errors="coerce")

        stats = quantity_stats.setdefault(
            column,
            {
                "non_missing": 0,
                "nonnumeric": 0,
                "zero": 0,
                "negative": 0,
                "positive": 0
            }
        )

        stats["non_missing"] += int(chunk[column].notna().sum())

        stats["nonnumeric"] += int(
            (chunk[column].notna() & values.isna()).sum()
        )

        stats["zero"] += int((values == 0).sum())

        stats["negative"] += int((values < 0).sum())

        stats["positive"] += int((values > 0).sum())

    # ---------------------------
    # Exact duplicate detection
    # ---------------------------

    row_hashes = pd.util.hash_pandas_object(
        chunk,
        index=False
    ).astype("uint64")

    unique_row_hashes = row_hashes.unique()

    # Convert hashes to hexadecimal strings
    row_hash_text = [
        (f"{int(h):016x}",)
        for h in unique_row_hashes
    ]

    before = conn.total_changes

    cursor.executemany(
        "INSERT OR IGNORE INTO full_rows VALUES (?)",
        row_hash_text
    )

    new_unique_rows = conn.total_changes - before

    exact_duplicates = len(chunk) - new_unique_rows

    exact_duplicate_count += exact_duplicates

    # ---------------------------
    # Duplicate medicine-month keys
    # ---------------------------

    valid_keys = chunk[identifier_columns].dropna()

    key_hashes = pd.util.hash_pandas_object(
        valid_keys,
        index=False
    ).astype("uint64")

    unique_key_hashes = key_hashes.unique()

    # Convert hashes to hexadecimal strings
    key_hash_text = [
        (f"{int(h):016x}",)
        for h in unique_key_hashes
    ]

    before = conn.total_changes

    cursor.executemany(
        "INSERT OR IGNORE INTO medicine_month_keys VALUES (?)",
        key_hash_text
    )

    new_unique_keys = conn.total_changes - before

    duplicate_keys = len(valid_keys) - new_unique_keys

    duplicate_key_count += duplicate_keys

    # ---------------------------
    # Save changes to SQLite
    # ---------------------------

    conn.commit()

# -------------------------------
# 5. Generate reports
# -------------------------------

# Monthly coverage report
monthly_report = pd.DataFrame(
    sorted(monthly_counts.items()),
    columns=["YEAR_MONTH", "RECORD_COUNT"]
)

monthly_report.to_csv(
    os.path.join(report_folder, "monthly_coverage.csv"),
    index=False
)

# Missing values report
missing_report = pd.DataFrame([
    {
        "COLUMN": column,
        "MISSING_COUNT": missing,
        "MISSING_PERCENT": round(
            missing / total_rows * 100, 2
        )
    }
    for column, missing in missing_counts.items()
])

missing_report.to_csv(
    os.path.join(report_folder, "missing_values.csv"),
    index=False
)

# Quantity analysis report
quantity_report = pd.DataFrame([
    {
        "COLUMN": column,
        **stats
    }
    for column, stats in quantity_stats.items()
])

quantity_report.to_csv(
    os.path.join(report_folder, "quantity_analysis.csv"),
    index=False
)

# -------------------------------
# 6. Duplicate summary
# -------------------------------

cursor.execute("SELECT COUNT(*) FROM full_rows")
unique_full_rows = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM medicine_month_keys")
unique_medicine_month_keys = cursor.fetchone()[0]

summary_file = os.path.join(
    report_folder, "audit_summary.txt"
)

with open(summary_file, "w") as f:

    f.write("DATA QUALITY AUDIT SUMMARY\n")
    f.write("======================= \n\n")

    f.write(f"Total records: {total_rows}\n")

    f.write(f"Unique exact rows: {unique_full_rows}\n")

    f.write(
        f"Exact duplicate rows: {exact_duplicate_count}\n"
    )

    f.write(
        f"Unique medicine-month-provider keys: "
        f"{unique_medicine_month_keys}\n"
    )

    f.write(
        f"Duplicate medicine-month-provider records: "
        f"{duplicate_key_count}\n"
    )

    f.write("\nMonthly coverage:\n")

    for month, count in sorted(monthly_counts.items()):
        f.write(f"{month}: {count}\n")

    f.write("\nMissing values:\n")

    for column, count in missing_counts.items():
        f.write(f"{column}: {count}\n")

    f.write("\nQuantity analysis:\n")

    for column, stats in quantity_stats.items():

        f.write(f"\n{column}\n")

        for metric, value in stats.items():
            f.write(f"  {metric}: {value}\n")

# -------------------------------
# 7. Close and remove temporary DB
# -------------------------------

conn.close()

if os.path.exists(db_path):
    os.remove(db_path)

# -------------------------------
# 8. Final output
# -------------------------------

print("\nData quality analysis completed!")

print("Total records:", total_rows)
print("Unique exact rows:", unique_full_rows)
print("Exact duplicate rows:", exact_duplicate_count)
print("Unique medicine-month-provider keys:", unique_medicine_month_keys)
print("Duplicate medicine-month-provider records:", duplicate_key_count)

print("\nReports saved in:")
print(report_folder)
