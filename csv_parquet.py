import pandas as pd

input_file = r"C:\Users\paula\OneDrive\Desktop\SETPart2\data\provisional_secondary_care_medicines_data.csv"

output_file = r"C:\Users\paula\OneDrive\Desktop\SETPart2\data\provisional_secondary_care_medicines_data.parquet"

df = pd.read_csv(input_file)

df.to_parquet(output_file, index=False)

print("Parquet conversion completed!")