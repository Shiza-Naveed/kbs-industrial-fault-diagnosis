"""Feature construction shared by training and evaluation (NaN-safe, recomputed after perturbation)."""
import numpy as np, pandas as pd
from .config import STRAIN_LIMIT

SENSOR_COLS = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
RAW_FEATURES = ["type_ord", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
ENG_FEATURES = RAW_FEATURES + ["delta_t", "power_w", "strain_ratio"]

def make_features(df):
    X = pd.DataFrame(index=df.index)
    X["type_ord"] = df["type"].map({"L": 0, "M": 1, "H": 2}).astype(float)
    for c in SENSOR_COLS: X[c] = df[c].astype(float)
    X["delta_t"] = X["process_temp_k"] - X["air_temp_k"]
    X["power_w"] = X["torque_nm"] * X["rpm"] * 2 * np.pi / 60
    X["strain_ratio"] = X["tool_wear_min"] * X["torque_nm"] / df["type"].map(STRAIN_LIMIT)
    return X

def perturb(df, noise_scale=0.0, missing_rate=0.0, seed=0):
    """Return a copy of df with Gaussian sensor noise and/or random missing sensor values."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    sigma = {"air_temp_k": 0.5, "process_temp_k": 0.5, "rpm": 25.0, "torque_nm": 1.5, "tool_wear_min": 3.0}
    for c in SENSOR_COLS:
        v = out[c].astype(float).values.copy()
        if noise_scale > 0: v = v + rng.normal(0, sigma[c] * noise_scale, len(v))
        if missing_rate > 0: v[rng.random(len(v)) < missing_rate] = np.nan
        out[c] = v
    return out
