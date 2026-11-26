import pandas as pd
import os
import glob

folder_path = r"C:\Users\paula\OneDrive\Desktop\SETPart2\data"

output_file = os.path.join(
    folder_path,
    "combined_secondary_care_medicines.csv"
)

# Only select original monthly files
files = sorted(
    glob.glob(os.path.join(folder_path, "scmd_provisional_*.csv"))
)

# Get all unique column names across monthly files
all_columns = []

for file in files:
    columns = pd.read_csv(file, nrows=0).columns.tolist()

    for col in columns:
        if col not in all_columns:
            all_columns.append(col)

# Remove previous output if it exists
if os.path.exists(output_file):
    os.remove(output_file)

first_chunk = True

# Process files in chunks
for file in files:
    print("Processing:", os.path.basename(file))

    for chunk in pd.read_csv(file, chunksize=50000, engine="python"):

        # Align all monthly data to the unified column structure
        chunk = chunk.reindex(columns=all_columns)

        chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk
        )

        first_chunk = False

print("\nCombination completed successfully!")
print("Monthly files:", len(files))
print("Unified columns:", len(all_columns))
print("Saved to:", output_file)