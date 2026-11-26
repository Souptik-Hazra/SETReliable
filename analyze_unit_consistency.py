"""Audit unit-name variation within the actual Trust/medicine/unit identity."""
import pandas as pd
from config import INPUT_FILE, CHUNK_SIZE


def analyze(path=INPUT_FILE, chunksize=CHUNK_SIZE):
    sets = {}
    names = {}
    for chunk in pd.read_csv(path, dtype="string", chunksize=chunksize):
        required = {"ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER",
                    "UNIT_OF_MEASURE_NAME", "VMP_PRODUCT_NAME"}
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing current-schema columns: {sorted(missing)}")
        for col in required:
            chunk[col] = chunk[col].str.strip()
        valid = chunk.dropna(subset=["ODS_CODE", "VMP_SNOMED_CODE"])
        for row in valid[["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER",
                          "UNIT_OF_MEASURE_NAME", "VMP_PRODUCT_NAME"]].itertuples(
                              index=False, name=None):
            trust, vmp, unit_id, unit_name, product = row
            key = (trust, vmp, unit_id if pd.notna(unit_id) else "UNKNOWN_UNIT_ID")
            sets.setdefault(key, set())
            if pd.notna(unit_name) and unit_name:
                sets[key].add(unit_name)
            names[key] = product
    rows = [{"ODS_CODE": k[0], "VMP_SNOMED_CODE": k[1],
             "UNIT_OF_MEASURE_IDENTIFIER": k[2], "VMP_PRODUCT_NAME": names[k],
             "UNIT_NAME_VARIANTS": len(v), "UNIT_NAMES": " | ".join(sorted(v))}
            for k, v in sets.items()]
    return pd.DataFrame(rows).sort_values(
        ["UNIT_NAME_VARIANTS", "ODS_CODE", "VMP_SNOMED_CODE"], ascending=[False, True, True])


if __name__ == "__main__":
    result = analyze()
    print(result.head(100).to_string(index=False))
    print(f"\nSeries with multiple unit names: {(result['UNIT_NAME_VARIANTS'] > 1).sum():,}")
