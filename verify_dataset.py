import pandas as pd
import os

folder_path = r"C:\Users\paula\OneDrive\Desktop\SETPart2\data"

file_path = os.path.join(
    folder_path,
    "combined_secondary_care_medicines.csv"
)

print("File exists:", os.path.exists(file_path))
print("File size (MB):", round(os.path.getsize(file_path) / (1024 * 1024), 2))

# Read column names
columns = pd.read_csv(file_path, nrows=0).columns.tolist()

print("\nColumns:")
print(columns)

# Initialize counters
total_rows = 0
monthly_counts = {}
missing_counts = {}

# Read in chunks to avoid memory problems
for chunk in pd.read_csv(file_path, chunksize=100000):

    total_rows += len(chunk)

    # Count records for each month
    month_counts = chunk["YEAR_MONTH"].value_counts()

    for month, count in month_counts.items():
        monthly_counts[month] = monthly_counts.get(month, 0) + count

    # Count missing values
    missing = chunk.isnull().sum()

    for column, count in missing.items():
        missing_counts[column] = missing_counts.get(column, 0) + count

# Display results
print("\nTotal records:", total_rows)

print("\nRecords per month:")
for month in sorted(monthly_counts):
    print(month, ":", monthly_counts[month])

print("\nMissing values:")
for column in columns:
    print(column, ":", missing_counts.get(column, 0))