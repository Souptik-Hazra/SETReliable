from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = DATA_DIR / "provisional_secondary_care_medicines_data.csv"

OUTPUT_BASE_FILE = OUTPUT_DIR / "simulated_base_data.csv"
OUTPUT_SUMMARY_FILE = OUTPUT_DIR / "simulation_summary.csv"

OUTPUT_ABRUPT_FILE = OUTPUT_DIR / "scenario_abrupt_surge.csv"
OUTPUT_GRADUAL_FILE = OUTPUT_DIR / "scenario_gradual_drift.csv"
OUTPUT_VOLATILITY_FILE = OUTPUT_DIR / "scenario_volatility_increase.csv"

# Columns expected in the NHSBSA provisional SCMD file
YEAR_MONTH_COL = "YEAR_MONTH"
ODS_CODE_COL = "ODS_CODE"
MEDICINE_CODE_COL = "VMP_SNOMED_CODE"
MEDICINE_NAME_COL = "VMP_PRODUCT_NAME"
QUANTITY_COL = "TOTAL_QUANTITY_IN_VMP_UDFS_UNIT_OF_MEASURE"
UNIT_COL = "VMP_UDFS_UNIT_OF_MEASURE_NAME"
COST_COL = "INDICATIVE_COST"

# Minimum number of monthly observations required per medicine-hospital series
MIN_HISTORY_MONTHS = 18

# Inventory simulation settings
INITIAL_INVENTORY_MULTIPLIER = 1.5
TARGET_STOCK_MULTIPLIER = 1.5
REORDER_QUANTILE = 0.90
RISK_ALERT_THRESHOLD = 0.20

# Reliability monitoring settings
ROLLING_WINDOW = 6
NOMINAL_COVERAGE = 0.90

# Reliability-score weights
WEIGHT_MAE = 0.35
WEIGHT_BIAS = 0.25
WEIGHT_CALIBRATION = 0.25
WEIGHT_DRIFT = 0.15

# Demand scenario settings
ABRUPT_SURGE_FACTORS = [1.5, 2.0, 3.0]
ABRUPT_SURGE_MONTHS = 3

GRADUAL_DRIFT_TOTAL_INCREASES = [0.20, 0.50, 1.00]
GRADUAL_DRIFT_MONTHS = 6

VOLATILITY_MULTIPLIERS = [1.5, 2.0, 3.0]

LEAD_TIME_DELAY_MONTHS = 1

RANDOM_SEED = 42