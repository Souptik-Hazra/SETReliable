import pandas as pd
import os
import glob

folder_path = r"C:\Users\paula\OneDrive\Desktop\SETPart2\data"

files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

reference_columns = None

for file in files:

    # Read only the header
    columns = pd.read_csv(file, nrows=0).columns.tolist()

    print("\nFile:", os.path.basename(file))
    print("Number of columns:", len(columns))
    print("Columns:", columns)

    if reference_columns is None:
        reference_columns = columns

    elif columns != reference_columns:
        print("⚠️ HEADER MISMATCH DETECTED!")

        print("Missing columns:", set(reference_columns) - set(columns))
        print("Extra columns:", set(columns) - set(reference_columns))

    else:
        print("Header matches.")

print("\nHeader checking completed.")
print("Total CSV files checked:", len(files))