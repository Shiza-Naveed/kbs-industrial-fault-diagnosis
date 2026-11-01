"""
STEP 4 - Enrich the cleaned data with SIMULATED operator observations and a train/test split.

The AI4I data contains only sensors. The problem statement requires handling incomplete and
conflicting evidence from operators, so observations are simulated (seeded, documented):
  * failing rows : 55 % -> one consistent symptom, 10 % -> one MISLEADING symptom, 35 % -> nothing reported
  * healthy rows : 3 %  -> one spurious symptom (false alarm from the operator)
Absence of an observation means "nothing reported" (unknown), not "nothing wrong".
Output: data/processed/ai4i2020_kbs_enriched.csv
"""
import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from kbs.config import *

SYMPTOM_OF = {"TWF": ["S_NOISE", "S_FINISH", "S_VIB"], "HDF": ["S_HOT", "S_COOLANT_LOW"],
              "PWF": ["S_BURN", "S_TRIP"], "OSF": ["S_STALL", "S_VIB"], "RNF": []}
ALL_SYMPTOMS = sorted({s for v in SYMPTOM_OF.values() for s in v})

def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA_PROC / "ai4i2020_clean.csv")
    obs, kind = [], []
    for _, r in df.iterrows():
        active = [f for f in FAULTS if r[f.lower()] == 1 and SYMPTOM_OF[f]]
        u = rng.random()
        if r["machine_failure"] == 1:
            if active and u < 0.55:
                f = rng.choice(active); obs.append(rng.choice(SYMPTOM_OF[f])); kind.append("consistent")
            elif u < 0.65:
                true_syms = {s for f in active for s in SYMPTOM_OF[f]}
                pool = [s for s in ALL_SYMPTOMS if s not in true_syms]
                obs.append(rng.choice(pool)); kind.append("misleading")
            else:
                obs.append(""); kind.append("none")
        else:
            if u < 0.03:
                obs.append(rng.choice(ALL_SYMPTOMS)); kind.append("spurious")
            else:
                obs.append(""); kind.append("none")
    df["observations"] = obs
    df["observation_kind"] = kind        # ground truth of how the observation was generated (for analysis only)
    strat = df["machine_failure"]
    tr, te = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=strat)
    df["split"] = "train"; df.loc[te, "split"] = "test"
    df.to_csv(DATA_PROC / "ai4i2020_kbs_enriched.csv", index=False)
    summ = {"rows": len(df), "train": int((df.split == "train").sum()), "test": int((df.split == "test").sum()),
            "observation_kind_counts": df.observation_kind.value_counts().to_dict(),
            "test_failures": int(df[df.split == "test"].machine_failure.sum())}
    (RESULTS / "enrichment_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
