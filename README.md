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

NHSBSA's August 2026 change log introduced renamed UDFS unit columns and additional dose-unit columns with the June 2026 provisional release. Those later columns are not assumed by this pipeline because they are not present in the current file.

## Quantity interpretation

NHSBSA clarification states that SCMD quantities are based on stock movements and stock-ledger balances. Negative values may represent returns or corrections and are usually corrected in finalised files; researchers must document their own cleaning protocol. The project therefore preserves three separate monthly measures:

- `GROSS_POSITIVE_QUANTITY`: sum of positive source quantity rows.
- `NEGATIVE_QUANTITY_TOTAL`: sum of negative source quantity rows, retaining the negative sign.
- `NET_QUANTITY`: signed sum of all source quantity rows.

Neither gross positive nor net quantity is automatically labelled patient-level demand. Both are forecast separately and compared using chronological one-step-ahead baseline evaluation. The negative movement measure remains available for audit and sensitivity analysis.

## Series identity and data quality

Monthly series are keyed by Trust (`ODS_CODE`), VMP SNOMED code, and unit-of-measure identifier. Unit name is descriptive metadata; changing names do not split a series. Different unit identifiers are never combined. Missing months remain missing, not zero. The loader attaches a `quality_report` to the returned DataFrame and the pipeline writes it to `outputs/data_quality_summary.csv`.

Indicative cost is retained as supplied, including negative values. Cost anomalies should be investigated before cost-based conclusions. Indicative cost is not necessarily a Trust's actual net procurement cost.

## Inventory simulation boundaries

Inventory simulation uses gross positive quantities as the issue/consumption stream. Optionally, the magnitude of negative movements is modelled as a separate return/adjustment inflow. This is an explicit sensitivity assumption, not proof that every negative entry is a physical return. Opening stock and supplier lead times are assumed because the dataset does not provide verified on-hand balances or delivery records. Shortages, stockouts and orders are therefore hypothetical outputs, not observed NHS Trust shortages or procurement events.

Synthetic surge, drift and volatility scenarios are stress tests and do not represent observed future outcomes. VEN classes are not fabricated; they remain unclassified unless a verified external mapping is supplied.

## Setup

Python 3.10+:

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

## Outputs

- `data_quality_summary.csv`: source-row and cleaning counts.
- `monthly_observed_measures.csv`: gross, negative and net monthly measures.
- `forecast_gross_positive_quantity.csv` and `forecast_net_quantity.csv`: separate target evaluations.
- `forecast_comparison.csv`: MAE, RMSE, WAPE and bias comparison.
- `inventory_*.csv`: hypothetical gross-issue inventory stress tests.
- `inventory_scenario_summary.csv`: synthetic scenario summary.

The forecast is a naive previous-observed-month baseline, not a trained forecasting model. The reliability indicators describe this baseline's historical out-of-sample errors and should not be presented as validation of a production forecasting system.


## Restored research workflow

The original SETReliable research flow is retained: data-quality investigation, medicine–Trust time-series construction, temporal feature engineering, demand stress scenarios, hypothetical inventory simulation, and reliability/drift monitoring. The current SCMD adapter is the only source-schema contract; older dataset column names are not accepted.

### Reproducible analysis commands

```bash
python dataset_profile.py
python data_quality_analysis.py
python analyze_unit_consistency.py
python investigate_quantities.py
python create_modeling_dataset.py
python run_simulation.py
pytest -q
```

The quality audit uses a temporary SQLite database for global exact-row and Trust–medicine–unit–month duplicate detection; it removes the database on normal completion. Pass an explicit `sqlite_path` to `audit()` when retaining the audit database is desired.

The modelling dataset includes calendar features, lags 1/2/3/6/12, and past-only rolling mean, standard deviation and maximum over 3/6 months. No future values are used in these features. Missing calendar months are marked as unobserved and are never silently converted into zero issue quantities.

Reliability monitoring includes rolling MAE/WAPE/bias, empirical prediction-interval coverage, CUSUM-style drift flags, and a lightweight adaptive-window mean-shift heuristic. The latter is explicitly an ADWIN-like heuristic, not a formal ADWIN implementation. These measures evaluate a naive previous-observation baseline, not a trained forecasting model.

VEN/clinical priority remains unclassified unless a verified mapping is supplied. The SCMD file does not contain enough information to infer clinical criticality or substitutability.
