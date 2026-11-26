# SETReliable — NHSBSA Secondary-Care Medicines

This branch adapts the project to the NHSBSA secondary-care medicines extract.

## Input data

Place the source file at:

```text
dataset/combined_secondary_care_medicines.csv
```

Expected columns (spelling is significant):

- `YEAR_MONTH` (YYYYMM)
- `ODS_CODE`
- `VMP_SNOMED_CODE`
- `VMP_PRODUCT_NAME`
- `UNIT_OF_MEASURE_IDENTIFIER`
- `UNIT_OF_MEASURE_NAME`
- `TOTAL_QUANITY_IN_VMP_UNIT`
- `INDICATIVE_COST`

The dataset is not committed. `dataset/`, generated `outputs/`, and local quality reports are ignored by Git.

## Environment and run

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python run_simulation.py
```

Outputs are written to `outputs/`.

## Data interpretation and limitations

- Rows are aggregated by month, Trust, VMP SNOMED code, unit identifier, and unit name. Different units are never added together.
- `SIGNED_NET_QUANTITY` retains the signed total. Negative source rows are counted and preserved as adjustment evidence.
- `DEMAND` is the sum of positive quantity rows only. This is a modelling convention pending authoritative confirmation of the meaning of negative entries; it must not be interpreted as validated net consumption.
- Missing Trust-medicine-unit months are represented as missing observations, not zero demand.
- Indicative cost is retained separately. Cost anomalies (including the reported 202502 spike) require source-level investigation before cost-based conclusions.
- Lead times and opening inventory are assumptions, not observed procurement/inventory records. Opening inventory defaults to zero to avoid future-data leakage.
- Abrupt surge, gradual drift, and volatility outputs are synthetic stress scenarios, not observed future outcomes.
- This dataset does not by itself provide on-hand stock, orders, supplier lead times, clinical VEN classes, or verified shortage outcomes. The simulation is therefore an assumption-driven scenario model, not a validated inventory/reliability forecast.

## Project modules

- `data_loader.py`: chunked CSV read and monthly aggregation
- `feature_engineering.py`: calendar, lag, rolling, and intermittent-demand features
- `demand_scenarios.py`: synthetic stress scenarios
- `inventory_simulation.py`: assumed-lead-time inventory simulation
- `reliability_monitoring.py`: historical residual monitoring and interval diagnostics
- `run_simulation.py`: end-to-end execution and output validation
- `tests/`: synthetic unit tests; no private dataset is required
