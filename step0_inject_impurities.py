"""
STEP 0 - Create the impure RAW working dataset.

The original AI4I 2020 file from Kaggle is clean (no missing values, no duplicates). To exercise a realistic
data-wrangling stage, known defects are deliberately injected into a COPY with a fixed seed. The untouched original
is kept as ground truth so the cleaning can be verified (step2b). Every injected defect is counted in
results/injection_manifest.json and disclosed in the report.
Input : data/raw/ai4i2020_original.csv      Output: data/raw/ai4i2020_impure.csv
"""
import json, numpy as np, pandas as pd
from kbs.config import *
from step2_clean_data import RENAME, SENSORS

def main():
    rng = np.random.default_rng(SEED)
    orig = pd.read_csv(ORIGINAL_RAW)
    audit = {"rows": int(len(orig)), "missing_values": int(orig.isna().sum().sum()), "duplicate_rows": int(orig.duplicated().sum()),
             "duplicate_udi": int(orig["UDI"].duplicated().sum())}
    d = orig.copy().astype({"Type": object}); inv = {v: k for k, v in RENAME.items()}; scol = [inv[c] for c in SENSORS]
    free = list(rng.permutation(len(d))); take = lambda n: [free.pop() for _ in range(n)]
    inj = {}
    r = take(8);  d.loc[r, "TWF"] = np.nan; inj["rows_with_missing_failure_label"] = 8
    r = take(60); dup_rows = d.loc[r].copy(); inj["exact_duplicate_rows"] = 60
    r = take(30); udi_dup = d.loc[r].copy(); udi_dup["Torque [Nm]"] += 5.0; inj["duplicate_udi_with_different_values"] = 30
    r = take(25); d.loc[r, "Type"] = [x.lower() + " " for x in d.loc[r, "Type"]]; inj["type_wrong_case_or_whitespace"] = 25
    r = take(15); d.loc[r, "Type"] = np.nan; inj["type_missing_recoverable_from_product_id"] = 15
    r = take(12)
    for i in r:
        c = rng.choice(scol); d[c] = d[c].astype(object); d.loc[i, c] = rng.choice(["n/a", "?", "error"])
    inj["non_numeric_sensor_values"] = 12
    r = take(15)
    for i in r:
        c = rng.choice(scol); d[c] = d[c].astype(object)
        d.loc[i, c] = {"Rotational speed [rpm]": -50, "Torque [Nm]": 999, "Tool wear [min]": 5000}.get(c, 0)
    inj["physically_impossible_sensor_values"] = 15
    r = take(120)
    for i in r:
        c = rng.choice(scol); d[c] = d[c].astype(object); d.loc[i, c] = np.nan
    inj["missing_sensor_values"] = 120
    dirty = pd.concat([d, dup_rows, udi_dup]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    dirty.to_csv(IMPURE_RAW, index=False)
    (RESULTS / "injection_manifest.json").write_text(json.dumps({"original_audit": audit, "injected": inj, "rows_after_injection": int(len(dirty)), "seed": SEED}, indent=2))
    print("original audit:", audit); print("injected:", inj); print("impure rows:", len(dirty), "->", IMPURE_RAW)

if __name__ == "__main__":
    main()
