"""
STEP 5 - Knowledge refinement: calibrate certainty factors of SENSOR rules from historical records.

The expert prior CF of each threshold rule is updated with its empirical precision on the TRAIN split
(P(fault | rule fires, not overridden)) using Bayesian shrinkage toward the expert prior:
        CF = (positives + K * prior) / (support + K),   K = 10 pseudo-observations
so rules with little historical support stay close to the expert estimate and well-supported rules follow the data.
Operator-observation and combined rules are calibrated the same way against the (simulated) observation history of the
training split, so single unverified operator reports are not over-trusted; disconfirming (negative) rules keep their expert CFs.
The TEST split is never touched here. (The expert-prior rules stay in rules.json and are evaluated as 'KBS-prior'.)
"""
import json, numpy as np, pandas as pd
from kbs.config import *

def mask(rule, df):
    m = pd.Series(True, index=df.index)
    for c in rule["conditions"]:
        if c["fact"] == "obs":
            m &= df["observations"].fillna("").str.split(";").apply(lambda L: c["value"] in L); continue
        col, v = df[c["fact"]], c["value"]
        m &= {"<": col < v, "<=": col <= v, ">": col > v, ">=": col >= v}[c["op"]]
    return m

def main():
    df = pd.read_csv(DATA_PROC / "ai4i2020_kbs_enriched.csv")
    tr = df[df.split == "train"]
    if not FIT_ON_IMPUTED: tr = tr[tr["imputed_cols"].fillna("") == ""]      # exclude imputed rows from calibration
    tr = tr.reset_index(drop=True)
    rules = json.loads((KB_DIR / "rules.json").read_text())
    by_id = {r["id"]: r for r in rules}
    CAL_KINDS = ("sensor", "observation", "combined")
    sensor = {r["id"]: r for r in rules if r["kind"] in CAL_KINDS}
    rows = []
    for r in rules:
        if r["kind"] not in CAL_KINDS:
            continue
        m = mask(r, tr)
        for o in sensor.values():                       # exclude rows handled by a more specific (overriding) rule
            if r["id"] in o["overrides"]: m &= ~mask(o, tr)
        n, pos = int(m.sum()), int(tr.loc[m, r["fault"].lower()].sum())
        prec = pos / n if n else float("nan")
        prior = r["cf"]
        K = 10
        r["cf"] = float(np.clip(round((pos + K * prior) / (n + K), 2), 0.05, 0.99))
        rows.append(dict(rule=r["id"], kind=r["kind"], fault=r["fault"], prior_cf=prior, support=n, positives=pos,
                         precision=round(prec, 3) if n else None, calibrated_cf=r["cf"]))
        r["prior_cf"] = prior; r["support"] = n; r["precision_train"] = round(prec, 3) if n else None
    (KB_DIR / "rules_calibrated.json").write_text(json.dumps(rules, indent=2))
    rep = pd.DataFrame(rows); rep.to_csv(RESULTS / "calibration_report.csv", index=False)
    print(rep.to_string(index=False))

if __name__ == "__main__":
    main()
