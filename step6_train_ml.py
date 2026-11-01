"""
STEP 6 - Train the machine-learning baselines used to ASSESS alternatives to the rule-based KBS.

Models   : Decision Tree, Random Forest (raw features and engineered features), Logistic Regression (engineered)
Targets  : machine_failure, TWF, HDF, PWF, OSF  (one binary classifier each; RNF is random by design)
Training : TRAIN split only, 5-fold stratified CV for hyper-parameter selection (decision-tree depth)
"""
import joblib, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from kbs.config import *
from kbs.features import make_features, RAW_FEATURES, ENG_FEATURES

TARGETS = ["machine_failure", "twf", "hdf", "pwf", "osf"]

def build(kind):
    imp = SimpleImputer(strategy="median")
    if kind == "dt":
        return GridSearchCV(Pipeline([("imp", imp), ("m", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))]),
                            {"m__max_depth": [3, 4, 5, 6, 8, 12], "m__min_samples_leaf": [1, 5, 20]},
                            scoring="average_precision", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    if kind == "rf":
        return Pipeline([("imp", imp), ("m", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                                                                      n_jobs=-1, random_state=SEED))])
    if kind == "lr":
        return Pipeline([("imp", imp), ("sc", StandardScaler()), ("m", LogisticRegression(class_weight="balanced", max_iter=2000))])

def main():
    df = pd.read_csv(DATA_PROC / "ai4i2020_kbs_enriched.csv")
    tr = df[df.split == "train"]
    if not FIT_ON_IMPUTED: tr = tr[tr["imputed_cols"].fillna("") == ""]      # exclude imputed rows from model fitting
    X = make_features(tr)
    specs = {"DT-raw": ("dt", RAW_FEATURES), "RF-raw": ("rf", RAW_FEATURES), "DT-eng": ("dt", ENG_FEATURES),
             "RF-eng": ("rf", ENG_FEATURES), "LR-eng": ("lr", ENG_FEATURES)}
    models, info = {}, []
    for name, (kind, feats) in specs.items():
        for t in TARGETS:
            m = build(kind); m.fit(X[feats], tr[t])
            best = getattr(m, "best_params_", None)
            if kind == "dt": m = m.best_estimator_
            models[(name, t)] = (m, feats)
            info.append(dict(model=name, target=t, best_params=json.dumps(best) if best else ""))
            print(f"trained {name:7s} -> {t}", best or "")
    joblib.dump(models, RESULTS / "ml_models.joblib")
    pd.DataFrame(info).to_csv(RESULTS / "ml_training_summary.csv", index=False)
    # interpretable decision-tree rules (explainability reference)
    out = []
    for t in ["hdf", "pwf", "osf", "twf"]:
        pipe, feats = models[("DT-eng", t)]
        out.append(f"=== Decision tree for {t.upper()} (depth {pipe.named_steps['m'].get_depth()}) ===\n" + export_text(pipe.named_steps["m"], feature_names=feats, max_depth=4))
    (RESULTS / "decision_tree_rules.txt").write_text("\n".join(out))

if __name__ == "__main__":
    main()
