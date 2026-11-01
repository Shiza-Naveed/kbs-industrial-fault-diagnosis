"""
STEP 2 - Data cleaning, validation and feature engineering (works on RAW data).

Input : data/raw/ai4i2020_impure.csv (original Kaggle file + injected impurities, step 0)
Output: data/processed/ai4i2020_clean.csv, results/cleaning_report.json, results/cleaning_log.csv

Cleaning policy (every action is counted in the report):
  1. normalise column names / strip whitespace; coerce sensor columns to numbers (non-numeric -> missing)
  2. remove exact duplicate rows; if a UDI key still appears with DIFFERENT values, drop all its rows (cannot tell which is right)
  3. product type: normalise case/spaces; if missing or invalid, recover it from the first letter of Product ID
  4. physically impossible sensor values (outside plausible ranges) -> set to missing (never silently kept)
  5. missing sensor values -> median of the same product type (flagged in 'imputed_cols'); rows with missing/invalid
     failure labels cannot be repaired and are dropped
  6. label consistency: failure modes are the primary labels.
       * any mode = 1 but Machine failure = 0  -> Machine failure set to 1 (a mode IS a failure)
       * Machine failure = 1 but no mode        -> kept as a failure and flagged 'unexplained_failure' = 1
         (real failures are not discarded; they cannot be diagnosed by any method)
  7. feature engineering: delta_t, power_w, strain, strain_limit, strain_ratio
"""
import json, numpy as np, pandas as pd
from kbs.config import *

RENAME = {
    "UDI": "udi", "Product ID": "product_id", "Type": "type",
    "Air temperature [K]": "air_temp_k", "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm", "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min", "Machine failure": "machine_failure",
    "TWF": "twf", "HDF": "hdf", "PWF": "pwf", "OSF": "osf", "RNF": "rnf",
}
SENSORS = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
MODES = ["twf", "hdf", "pwf", "osf", "rnf"]
RANGES = {"air_temp_k": (280, 330), "process_temp_k": (280, 340), "rpm": (300, 4000),
          "torque_nm": (0, 120), "tool_wear_min": (0, 300)}      # plausible physical ranges (domain assumption)

def clean(raw: pd.DataFrame):
    rep, df = {"rows_in": int(len(raw))}, raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=RENAME)
    missing_cols = [c for c in RENAME.values() if c not in df.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"
    df = df[list(RENAME.values())]

    # 1. numeric coercion
    nonnum = 0
    for c in SENSORS + ["udi", "machine_failure"] + MODES:
        before = df[c].isna().sum(); df[c] = pd.to_numeric(df[c], errors="coerce"); nonnum += int(df[c].isna().sum() - before)
    rep["non_numeric_values_set_missing"] = nonnum

    # 2. duplicates
    rep["exact_duplicate_rows_removed"] = int(df.duplicated().sum()); df = df.drop_duplicates()
    # duplicate keys whose values DIFFER cannot be resolved without the source system -> drop every copy (conservative)
    dup_key = df["udi"].duplicated(keep=False) & df["udi"].notna()
    rep["duplicate_udi_keys_with_conflicting_values"] = int(df.loc[dup_key, "udi"].nunique())
    rep["rows_dropped_conflicting_duplicate_keys"] = int(dup_key.sum()); df = df[~dup_key]

    # 3. product type
    t = df["type"].astype(str).str.strip().str.upper().replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    bad = ~t.isin(["L", "M", "H"])
    rep["type_normalised"] = int(((df["type"].astype(str) != t.astype(str)) & ~bad).sum())
    from_pid = df["product_id"].astype(str).str.strip().str.upper().str[0]
    rec = bad & from_pid.isin(["L", "M", "H"])
    t[rec] = from_pid[rec]; rep["type_recovered_from_product_id"] = int(rec.sum())
    rep["type_unrecoverable_dropped"] = int((~t.isin(["L", "M", "H"])).sum())
    df["type"] = t; df = df[df["type"].isin(["L", "M", "H"])]

    # 4. out-of-range -> missing
    oor = 0
    for c, (lo, hi) in RANGES.items():
        b = (df[c] < lo) | (df[c] > hi); rep[f"out_of_range_{c}"] = int(b.sum()); oor += int(b.sum()); df.loc[b, c] = np.nan
    rep["out_of_range_total_set_missing"] = oor

    # 5. missing values
    rep["missing_sensor_values_before_imputation"] = int(df[SENSORS].isna().sum().sum())
    lab_bad = df[MODES].isna().any(axis=1) | ~df[MODES].isin([0, 1]).all(axis=1)
    rep["rows_dropped_missing_or_invalid_failure_label"] = int(lab_bad.sum()); df = df[~lab_bad].copy()
    df["imputed_cols"] = ""
    for c in SENSORS:
        m = df[c].isna()
        if m.any():
            df.loc[m, c] = df.groupby("type")[c].transform("median")[m]
            df.loc[m, "imputed_cols"] += c + ";"
    rep["rows_with_imputed_values"] = int((df["imputed_cols"] != "").sum())
    rep["missing_values_after"] = int(df[SENSORS + ["type"]].isna().sum().sum())

    # 6. label consistency
    modes = df[MODES].astype(int); any_mode = modes.max(axis=1)
    mf = df["machine_failure"]
    fix_up = (mf.fillna(0) == 0) & (any_mode == 1)
    unexplained = (mf == 1) & (any_mode == 0)
    rep["label_conflict_mode_without_failure_repaired"] = int(fix_up.sum())
    rep["label_conflict_failure_without_mode_flagged"] = int(unexplained.sum())
    df["machine_failure"] = np.where(fix_up | (any_mode == 1) | (mf == 1), 1, 0)
    df["unexplained_failure"] = unexplained.astype(int)
    for c in MODES + ["udi"]: df[c] = df[c].astype(int)

    # 7. features
    df["delta_t"] = (df["process_temp_k"] - df["air_temp_k"]).round(2)
    df["power_w"] = (df["torque_nm"] * df["rpm"] * 2 * np.pi / 60).round(1)
    df["strain"] = (df["tool_wear_min"] * df["torque_nm"]).round(1)
    df["strain_limit"] = df["type"].map(STRAIN_LIMIT)
    df["strain_ratio"] = (df["strain"] / df["strain_limit"]).round(4)
    df = df.sort_values("udi").reset_index(drop=True)
    rep.update(rows_out=int(len(df)), failure_rate=round(float(df["machine_failure"].mean()), 4),
               failures=int(df["machine_failure"].sum()), mode_counts={m.upper(): int(df[m].sum()) for m in MODES},
               multi_fault_rows=int((df[MODES].sum(axis=1) > 1).sum()))
    return df, rep

def main():
    src = IMPURE_RAW if IMPURE_RAW.exists() else (ORIGINAL_RAW if ORIGINAL_RAW.exists() else REPLICA_RAW)
    raw = pd.read_csv(src)
    df, rep = clean(raw); rep["source_file"] = src.name
    rep["raw_missing_values"] = int(raw.isna().sum().sum()); rep["raw_duplicate_rows"] = int(raw.duplicated().sum())
    df.to_csv(DATA_PROC / "ai4i2020_clean.csv", index=False)
    (RESULTS / "cleaning_report.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame([{"step": k, "count": v} for k, v in rep.items() if isinstance(v, (int, float))]).to_csv(RESULTS / "cleaning_log.csv", index=False)
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
