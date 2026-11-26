"""Summarise signed quantity movements using the current SCMD schema."""
import pandas as pd
from config import INPUT_FILE, CHUNK_SIZE


def investigate(path=INPUT_FILE, chunksize=CHUNK_SIZE):
    monthly = {}
    samples = []
    for chunk in pd.read_csv(path, dtype={"YEAR_MONTH": "string",
            "ODS_CODE": "string", "VMP_SNOMED_CODE": "string"}, chunksize=chunksize):
        q = pd.to_numeric(chunk["TOTAL_QUANITY_IN_VMP_UNIT"], errors="coerce")
        work = pd.DataFrame({"YEAR_MONTH": chunk["YEAR_MONTH"], "quantity": q})
        for month, group in work.groupby("YEAR_MONTH", dropna=False):
            state = monthly.setdefault(str(month), {"rows": 0, "negative_rows": 0,
                "zero_rows": 0, "positive_rows": 0, "negative_total": 0.0})
            values = group["quantity"]
            state["rows"] += len(group)
            state["negative_rows"] += int((values < 0).sum())
            state["zero_rows"] += int((values == 0).sum())
            state["positive_rows"] += int((values > 0).sum())
            state["negative_total"] += float(values[values < 0].sum())
        neg = chunk.loc[q < 0, ["YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE",
                                "VMP_PRODUCT_NAME", "UNIT_OF_MEASURE_IDENTIFIER",
                                "UNIT_OF_MEASURE_NAME"]].copy()
        neg["QUANTITY"] = q[q < 0]
        if len(samples) < 30:
            samples.extend(neg.head(30 - len(samples)).to_dict("records"))
    return (pd.DataFrame.from_dict(monthly, orient="index").rename_axis("YEAR_MONTH")
            .reset_index().sort_values("YEAR_MONTH"), pd.DataFrame(samples))


if __name__ == "__main__":
    monthly, samples = investigate()
    print(monthly.to_string(index=False))
    print("\nRepresentative negative rows (not a semantic classification):")
    print(samples.to_string(index=False))
