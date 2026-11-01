"""Central configuration: paths, constants, random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
KB_DIR = ROOT / "knowledge_base"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

import os
SEED = 42
# Rows whose sensor values were imputed (flagged in step 2) no longer reflect the physical state that produced their label.
# By default they are excluded from calibration and model fitting (they stay in the test split). KBS_FIT_INCLUDE_IMPUTED=1 includes them (ablation).
FIT_ON_IMPUTED = os.environ.get("KBS_FIT_INCLUDE_IMPUTED", "0") == "1"
ORIGINAL_RAW = DATA_RAW / "ai4i2020_original.csv"          # untouched Kaggle file (clean) - ground truth
IMPURE_RAW = DATA_RAW / "ai4i2020_impure.csv"              # working raw data: original + deliberately injected impurities (step 0)
REAL_RAW = ORIGINAL_RAW
REPLICA_RAW = DATA_RAW / "ai4i2020_synthetic_replica.csv"  # generated stand-in

FAULTS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
PREDICTABLE_FAULTS = ["TWF", "HDF", "PWF", "OSF"]

# Belief (certainty factor) levels used for reporting a diagnosis
LEVEL_CONFIRMED = 0.70
LEVEL_PROBABLE = 0.40
LEVEL_WATCH = 0.10
DECISION_THRESHOLD = LEVEL_PROBABLE   # belief >= 0.40 counts as "fault alert" in evaluation

# Strain limits (tool_wear * torque, min*Nm) per product-quality type [Matzka 2020]
STRAIN_LIMIT = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
