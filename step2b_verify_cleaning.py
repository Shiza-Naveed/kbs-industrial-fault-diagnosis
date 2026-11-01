"""
STEP 2b - Verify the wrangling against the untouched original (ground truth).
Compares data/processed/ai4i2020_clean.csv (cleaned impure data) with the cleaned ORIGINAL file.
Output: results/cleaning_verification.json
"""
import json, numpy as np, pandas as pd
from kbs.config import *
from step2_clean_data import clean, SENSORS

def main():
    ref, _ = clean(pd.read_csv(ORIGINAL_RAW)); new = pd.read_csv(DATA_PROC / "ai4i2020_clean.csv")
    new["imputed_cols"] = new["imputed_cols"].fillna("")
    m = ref.merge(new, on="udi", suffixes=("_ref", "_new")); touched = m["imputed_cols_new"] != ""
    same = all((m[f"{c}_ref"] == m[f"{c}_new"]).loc[~touched].all() for c in SENSORS + ["type"])
    inj = json.loads((RESULTS / "injection_manifest.json").read_text())["injected"]
    expected = len(ref) - inj["rows_with_missing_failure_label"] - inj["duplicate_udi_with_different_values"]
    err = {c: float(np.nanmean(np.abs(m.loc[touched, f"{c}_ref"] - m.loc[touched, f"{c}_new"]))) for c in SENSORS}
    def agree(df):
        h = ((df.process_temp_k - df.air_temp_k) < 8.6) & (df.rpm < 1380); return float((h == df.hdf.astype(bool)).mean())
    res = {"rows_original_clean": int(len(ref)), "rows_after_cleaning": int(len(new)), "expected_rows": int(expected),
           "row_count_matches_expectation": bool(len(new) == expected), "untouched_rows_identical_to_original": bool(same),
           "rows_with_imputed_values": int(touched.sum()), "mean_abs_imputation_error": err,
           "hdf_rule_agreement_imputed_rows": agree(m.loc[touched, [c + "_new" for c in ["process_temp_k", "air_temp_k", "rpm", "hdf"]]].rename(columns=lambda c: c[:-4])),
           "hdf_rule_agreement_original_same_rows": agree(m.loc[touched, [c + "_ref" for c in ["process_temp_k", "air_temp_k", "rpm", "hdf"]]].rename(columns=lambda c: c[:-4]))}
    (RESULTS / "cleaning_verification.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
