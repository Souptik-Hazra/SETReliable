# SETReliable — NHSBSA Secondary-Care Medicines

## Dataset

Place the current source file at `dataset/combined_secondary_care_medicines.csv`.
The repository does not commit the dataset or generated outputs.

The current file covers September 2023–May 2026 and uses this pre-June-2026 schema:

- `YEAR_MONTH` (YYYYMM)
- `ODS_CODE`
- `VMP_SNOMED_CODE`
- `VMP_PRODUCT_NAME`
- `UNIT_OF_MEASURE_IDENTIFIER`
- `UNIT_OF_MEASURE_NAME`
- `TOTAL_QUANITY_IN_VMP_UNIT` (source spelling)
- `INDICATIVE_COST`

The August 2026 schema changes apply to the June 2026 provisional release and are not assumed here.

## Quantity interpretation

SCMD quantities are based on stock movements and stock-ledger balances. Negative values may represent returns or corrections; their precise row-level meaning is not established by this file. The pipeline preserves three separate monthly measures:

- `GROSS_POSITIVE_QUANTITY`: sum of positive quantity rows; zero when a source-observed month has no positive rows.
- `NEGATIVE_QUANTITY_TOTAL`: sum of negative quantity rows, retaining the negative sign; zero when a source-observed month has no negative rows.
- `NET_QUANTITY`: signed sum of all valid source quantity rows.

A month with source records and no positive/negative movements is a measured zero for the relevant component. A calendar month with no source records is marked `OBSERVATION_MISSING=1` and its quantity measures remain missing. These cases are never conflated.

Neither gross-positive nor net quantity is automatically patient-level demand. Both are movement measures and are evaluated separately. Indicative cost is not necessarily a Trust's actual net procurement cost.

## Series identity and data quality

Monthly series are keyed by Trust, VMP SNOMED code and unit-of-measure identifier. Missing unit identifiers are consistently assigned `UNKNOWN_UNIT_ID`; different unit identifiers are never combined. Unit names are descriptive metadata and name variants are reported.

The loader uses DuckDB to aggregate the CSV globally with a configurable memory limit and disk spill. Only the aggregated monthly result is materialised in pandas. This substantially reduces the memory burden compared with retaining every chunk's intermediate DataFrame, but peak memory and runtime still depend on machine resources and disk capacity. Validate on the target machine before assuming full-file execution will complete.

The global duplicate audit also uses DuckDB disk-backed grouping: exact full-row duplicates and duplicate Trust–VMP–unit–month keys are counted across the entire file, including chunk boundaries. The `chunksize` argument is retained for command compatibility but is not used by the DuckDB global query.

## Inventory simulation boundaries

SCMD does not provide verified opening stock balances, receipts, orders, supplier lead times or patient-level consumption. The inventory module is therefore a hypothetical stress-test of recorded gross-positive movements, not a reconstruction of actual Trust inventory or observed stockouts. Opening inventory and lead times are explicit assumptions. Negative movements are **not** treated as return inflows by default; enabling that behavior is an optional sensitivity assumption, not an established interpretation.

Outputs include `INVENTORY_SIMULATION_IS_HYPOTHETICAL=1` and `STOCKOUTS_ARE_OBSERVED=0`. Do not report simulated stockout rates as real NHS Trust stockout rates or supplier reliability.

## Research scope

The current 33-month dataset supports descriptive analysis of recorded medicine stock movements, unit-aware medicine–Trust time-series construction, retrospective feature engineering, chronological baseline evaluation, and synthetic surge/drift/volatility stress scenarios.

It does not independently establish patient-level demand, actual stock availability, real stockout events, replenishment performance or supplier reliability. Those questions require additional data such as stock-on-hand balances, procurement/receipt records, supplier lead times and/or dispensing/administration data.

VEN/clinical priority remains unclassified unless a verified external mapping is supplied. No clinical criticality or substitutability is inferred from medicine codes.

## Setup

Python 3.11:

```bash
python -m venv .venv
# Windows Git Bash
source .venv/Scripts/activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python run_simulation.py
```

## Analysis commands

```bash
python dataset_profile.py
python data_quality_analysis.py
python analyze_unit_consistency.py
python investigate_quantities.py
python create_modeling_dataset.py
python run_simulation.py
pytest -q
```

## Outputs

- `data_quality_summary.csv`: source-row and cleaning counts.
- `monthly_observed_measures.csv`: gross, negative and net monthly movement measures.
- `forecast_gross_positive_quantity.csv` and `forecast_net_quantity.csv`: separate baseline evaluations.
- `forecast_comparison.csv`: MAE, RMSE, WAPE and bias comparison.
- `inventory_*.csv`: explicitly hypothetical movement stress tests.
- `inventory_scenario_summary.csv`: synthetic scenario summaries.

The forecast is a naive previous-observed-month baseline, not a trained forecasting model. Reliability indicators evaluate this baseline's historical errors; they do not validate a production forecasting system.

The restored SETReliable research flow is retained: data-quality investigation, medicine–Trust time-series construction, temporal feature engineering, movement stress scenarios, hypothetical inventory simulation and reliability/drift monitoring. The current SCMD adapter is the only source-schema contract.
