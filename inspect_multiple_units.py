import pandas as pd
from pathlib import Path

file_path = Path(
    r"C:\Users\paula\OneDrive\Desktop\SETPart2\data\data_quality_reports\medicine_unit_consistency.csv"
)

df = pd.read_csv(file_path)

multiple_units = df[df["UNIT_STATUS"] == "Multiple units"]

print(multiple_units.to_string(index=False))