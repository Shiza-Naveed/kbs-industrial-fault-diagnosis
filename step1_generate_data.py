"""
STEP 1 - Build a schema-identical replica of the AI4I 2020 Predictive Maintenance dataset.

Why: the sandbox used to build this project has no internet access, so the original
Kaggle/UCI file could not be downloaded. The generator follows the failure physics
documented by the dataset author (Matzka, 2020):
  * product quality type L/M/H = 60/30/10 %, tool wear grows 2/3/5 min per part
  * process temperature = air temperature + ~10 K
  * HDF: (process - air temperature) < 8.6 K AND rpm < 1380
  * PWF: power = torque * omega  < 3500 W or > 9000 W
  * OSF: tool_wear * torque > 11000 / 12000 / 13000 minNm (L / M / H)
  * TWF: tool fails/is replaced at a random wear in [200, 240] min
  * RNF: 0.1 % random failure regardless of parameters
If the REAL file is placed at data/raw/ai4i2020.csv the remaining steps use it instead.
"""
import numpy as np, pandas as pd
from kbs.config import *

def generate(n=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    types = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    inc = {"L": 2, "M": 3, "H": 5}

    # slowly drifting ambient temperature (AR(1)) around 300 K, sd ~2 K
    air = np.zeros(n); air[0] = 298.5
    for i in range(1, n):
        air[i] = 300 + 0.995 * (air[i-1] - 300) + rng.normal(0, 0.2)
    air += rng.normal(0, 0.3, n)
    delta = 10.0 + rng.normal(0, 1.0, n)
    proc = air + delta

    rpm = 1300 + rng.gamma(1.8, 132, n) + rng.normal(0, 25, n)
    rpm = np.round(rpm).astype(int)
    torque = 40 - 0.038 * (rpm - 1538) + rng.normal(0, 4.0, n)
    torque = np.clip(torque, 3.0, None)

    wear = np.zeros(n, int); twf = np.zeros(n, int)
    w, thr = 0, rng.integers(200, 241)
    for i in range(n):
        w += inc[types[i]]
        wear[i] = w
        if w >= thr:
            if rng.random() < 0.45:
                twf[i] = 1
            w, thr = 0, rng.integers(200, 241)

    power = torque * rpm * 2 * np.pi / 60
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pwf = (power < 3500) | (power > 9000)
    lim = np.array([STRAIN_LIMIT[t] for t in types])
    osf = wear * torque > lim
    rnf = rng.random(n) < 0.0015
    fail = (twf | hdf | pwf | osf | rnf).astype(int)

    df = pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"{t}{47000 + i}" for i, t in enumerate(types)],
        "Type": types,
        "Air temperature [K]": air.round(1),
        "Process temperature [K]": proc.round(1),
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque.round(1),
        "Tool wear [min]": wear,
        "Machine failure": fail,
        "TWF": twf, "HDF": hdf.astype(int), "PWF": pwf.astype(int),
        "OSF": osf.astype(int), "RNF": rnf.astype(int),
    })
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv(REPLICA_RAW, index=False)
    print(f"Saved {REPLICA_RAW}  shape={df.shape}")
    print(f"failure rate = {df['Machine failure'].mean():.2%}")
    print(df[FAULTS].sum().to_string())
