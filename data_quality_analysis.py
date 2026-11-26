"""Chunked SCMD quality audit, including disk-backed global duplicate checks."""
import hashlib
import sqlite3
from pathlib import Path
import pandas as pd
from config import INPUT_FILE, CHUNK_SIZE
from data_loader import REQUIRED_COLUMNS


def audit(path=INPUT_FILE, chunksize=CHUNK_SIZE, sqlite_path=None):
    db_path = Path(sqlite_path) if sqlite_path else Path(".scmd_quality_audit.sqlite")
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE hashes (hash TEXT PRIMARY KEY)")
    conn.execute("""CREATE TABLE keys (
        year_month TEXT, ods TEXT, vmp TEXT, unit_id TEXT,
        PRIMARY KEY(year_month, ods, vmp, unit_id))""")
    summary = {"rows": 0, "missing_cost": 0, "negative_quantity": 0,
               "zero_quantity": 0, "negative_cost": 0, "invalid_quantity": 0,
               "invalid_month": 0, "exact_duplicate_rows": 0,
               "duplicate_series_month_keys": 0}
    try:
        for chunk in pd.read_csv(path, dtype="string", chunksize=chunksize, low_memory=False):
            missing = set(REQUIRED_COLUMNS) - set(chunk.columns)
            if missing:
                raise ValueError(f"Missing required columns: {sorted(missing)}")
            summary["rows"] += len(chunk)
            q = pd.to_numeric(chunk["TOTAL_QUANITY_IN_VMP_UNIT"], errors="coerce")
            c = pd.to_numeric(chunk["INDICATIVE_COST"], errors="coerce")
            summary["missing_cost"] += int(c.isna().sum())
            summary["negative_quantity"] += int((q < 0).sum())
            summary["zero_quantity"] += int((q == 0).sum())
            summary["negative_cost"] += int((c < 0).sum())
            summary["invalid_quantity"] += int(q.isna().sum())
            valid_month = pd.to_datetime(chunk["YEAR_MONTH"], format="%Y%m", errors="coerce")
            summary["invalid_month"] += int(valid_month.isna().sum())
            for row in chunk.itertuples(index=False, name=None):
                raw = "\x1f".join("" if pd.isna(v) else str(v) for v in row)
                digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                cur = conn.execute("INSERT OR IGNORE INTO hashes VALUES (?)", (digest,))
                if cur.rowcount == 0:
                    summary["exact_duplicate_rows"] += 1
            keys = chunk[["YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE",
                          "UNIT_OF_MEASURE_IDENTIFIER"]].dropna(
                              subset=["YEAR_MONTH", "ODS_CODE", "VMP_SNOMED_CODE"])
            for row in keys.itertuples(index=False, name=None):
                cur = conn.execute("INSERT OR IGNORE INTO keys VALUES (?,?,?,?)",
                                   tuple(str(v) for v in row))
                if cur.rowcount == 0:
                    summary["duplicate_series_month_keys"] += 1
            conn.commit()
    finally:
        conn.close()
        if sqlite_path is None and db_path.exists():
            db_path.unlink()
    return summary


if __name__ == "__main__":
    print(audit())
