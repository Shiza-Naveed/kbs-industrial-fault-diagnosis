"""
STEP 7 - Testing and evaluation on the held-out TEST split (2,000 records).

Compared approaches
  KBS-sensor : rule base using only sensor-threshold rules (forward chaining, certainty factors)
  KBS-full   : + operator-observation, combined and disconfirming (conflict-handling) rules
  DT / RF / LR : machine-learning baselines (raw or engineered features)
Experiments
  E1 clean test data              E2 sensor noise (x0.5, x1, x2)       E3 missing sensor values (5-30 %)
  E4 conflicting operator evidence (ablation of the disconfirming rules)
  E5 stakeholder-profile sensitivity      E6 timing, avoidable-cost model, explainability
"""
import json, time, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from kbs.config import *
from kbs.engine import KnowledgeBase, InferenceEngine
from kbs.recommender import recommend
from kbs.features import make_features, perturb, SENSOR_COLS

TARGETS = {"machine_failure": "Machine failure", "twf": "TWF", "hdf": "HDF", "pwf": "PWF", "osf": "OSF"}
FAULT4 = ["twf", "hdf", "pwf", "osf"]
METHODS = ["KBS-sensor", "KBS-full", "DT-raw", "RF-raw", "DT-eng", "RF-eng", "LR-eng"]
ACTION_COST = {"twf": 60, "hdf": 150, "pwf": 400, "osf": 300}
MISS_MULT = 5          # assumption: an unplanned breakdown costs 5x the planned action

KB_FULL = KnowledgeBase(); KB_SENS = KB_FULL.without(["observation", "combined", "disconfirming"])
KB_NODIS = KB_FULL.without(["disconfirming"])
KB_PRIOR = KnowledgeBase(calibrated=False)   # original expert-prior certainty factors (before calibration)
ENG = {"KBS-full": InferenceEngine(KB_FULL), "KBS-sensor": InferenceEngine(KB_SENS), "KBS-nodis": InferenceEngine(KB_NODIS), "KBS-prior": InferenceEngine(KB_PRIOR)}
MODELS = joblib.load(RESULTS / "ml_models.joblib")

def evidence(row):
    o = row.get("observations"); o = "" if (o is None or (isinstance(o, float) and np.isnan(o))) else o
    return {k: row[k] for k in ["type"] + SENSOR_COLS} | {"observations": o}

def kbs_scores(engine, d):
    """belief per predictable fault (clipped at 0) for every row of d."""
    B = {t: np.zeros(len(d)) for t in FAULT4}
    for i, row in enumerate(d.to_dict("records")):
        dg = engine.diagnose(evidence(row))
        for t in FAULT4: B[t][i] = max(0.0, dg.belief[t.upper()])
    B["machine_failure"] = np.max([B[t] for t in FAULT4], axis=0)
    return B

def ml_scores(name, d):
    X = make_features(d); S = {}
    for t in TARGETS:
        pipe, feats = MODELS[(name, t)]; S[t] = pipe.predict_proba(X[feats])[:, 1]
    return S

def all_scores(d):
    out = {"KBS-sensor": kbs_scores(ENG["KBS-sensor"], d), "KBS-full": kbs_scores(ENG["KBS-full"], d)}
    for m in METHODS[2:]: out[m] = ml_scores(m, d)
    return out

def thr(m): return DECISION_THRESHOLD if m.startswith("KBS") else 0.5

def metrics(y, s, t):
    yp = (s >= t).astype(int)
    return dict(precision=precision_score(y, yp, zero_division=0), recall=recall_score(y, yp, zero_division=0),
                f1=f1_score(y, yp, zero_division=0), pr_auc=average_precision_score(y, s),
                tp=int(((yp == 1) & (y == 1)).sum()), fp=int(((yp == 1) & (y == 0)).sum()), fn=int(((yp == 0) & (y == 1)).sum()))

def table(d, scores):
    rows = []
    for m in METHODS:
        for t, label in TARGETS.items():
            r = metrics(d[t].values, scores[m][t], thr(m)); r.update(method=m, target=label); rows.append(r)
    return pd.DataFrame(rows)

def macro_f1(d, scores, m, faults=FAULT4):
    return float(np.mean([f1_score(d[t].values, (scores[m][t] >= thr(m)).astype(int), zero_division=0) for t in faults]))

def main():
    df = pd.read_csv(DATA_PROC / "ai4i2020_kbs_enriched.csv")
    te = df[df.split == "test"].reset_index(drop=True)
    res = {"n_test": len(te), "test_failures": int(te.machine_failure.sum()),
           "test_mode_counts": {t.upper(): int(te[t].sum()) for t in FAULT4 + ["rnf"]}}

    # ---------------------------------------------------------------- E1 clean
    S = all_scores(te); T1 = table(te, S); T1.to_csv(RESULTS / "metrics_clean.csv", index=False)
    # top-1 diagnosis accuracy on failing rows with at least one known (non-RNF) mode
    known = te[(te[FAULT4].sum(axis=1) > 0)].index.values
    top1 = {}
    for m in METHODS:
        hit = 0
        for i in known:
            sc = {t: S[m][t][i] for t in FAULT4}; best = max(sc, key=sc.get)
            hit += int(sc[best] >= thr(m) and te.loc[i, best] == 1)
        top1[m] = hit / len(known)
    res["top1_accuracy"] = top1; res["n_top1_cases"] = int(len(known))
    # avoidable-cost model
    cost = {}
    for m in METHODS:
        c = 0
        for t in FAULT4:
            r = T1[(T1.method == m) & (T1.target == t.upper())].iloc[0]
            c += r.fp * ACTION_COST[t] + r.fn * MISS_MULT * ACTION_COST[t]
        cost[m] = float(c)
    res["avoidable_cost_usd"] = cost

    SP = kbs_scores(ENG["KBS-prior"], te); prior_rows = []
    for t, label in TARGETS.items():
        r = metrics(te[t].values, SP[t], DECISION_THRESHOLD); r.update(method="KBS-prior", target=label); prior_rows.append(r)
    PRI = pd.DataFrame(prior_rows); PRI.to_csv(RESULTS / "metrics_kbs_prior.csv", index=False)
    res["macro_f1_kbs_prior"] = float(np.mean([f1_score(te[t].values, (SP[t] >= DECISION_THRESHOLD).astype(int), zero_division=0) for t in FAULT4]))
    res["macro_f1_kbs_prior_no_twf"] = float(np.mean([f1_score(te[t].values, (SP[t] >= DECISION_THRESHOLD).astype(int), zero_division=0) for t in ["hdf", "pwf", "osf"]]))

    # ------------------------------------------------------ E2 noise, E3 missing
    rob = []
    for kind, levels in [("noise", [0, 0.5, 1, 2]), ("missing", [0, 0.05, 0.10, 0.20, 0.30])]:
        for lv in levels:
            d = perturb(te, noise_scale=lv if kind == "noise" else 0, missing_rate=lv if kind == "missing" else 0, seed=7)
            sc = all_scores(d)
            for m in METHODS:
                rob.append(dict(experiment=kind, level=lv, method=m, macro_f1=macro_f1(te, sc, m),
                                macro_f1_no_twf=macro_f1(te, sc, m, ["hdf", "pwf", "osf"]),
                                machine_failure_f1=f1_score(te.machine_failure, (sc[m]["machine_failure"] >= thr(m)).astype(int), zero_division=0)))
    R = pd.DataFrame(rob); R.to_csv(RESULTS / "metrics_robustness.csv", index=False)

    # ------------------------------------ E4 conflicting operator evidence
    obs = df[df.observation_kind != "none"].reset_index(drop=True)
    conf = []
    for grp, g in obs.groupby("observation_kind"):
        g = g.reset_index(drop=True); row = dict(group=grp, n=len(g))
        for name in ["KBS-sensor", "KBS-prior", "KBS-nodis", "KBS-full"]:
            B = kbs_scores(ENG[name], g)
            false_alert = np.zeros(len(g), bool); true_found = np.zeros(len(g), bool); false_watch = np.zeros(len(g), bool)
            for t in FAULT4:
                a = B[t] >= DECISION_THRESHOLD
                false_alert |= a & (g[t].values == 0); true_found |= a & (g[t].values == 1)
                false_watch |= (B[t] >= LEVEL_WATCH) & (g[t].values == 0)
            row[f"{name}_rows_with_false_watch"] = int(false_watch.sum())
            row[f"{name}_rows_with_false_alert"] = int(false_alert.sum()); row[f"{name}_true_fault_found"] = int(true_found.sum())
        row["operator_only_false_alert"] = int(len(g)) if grp in ("spurious",) else None
        conf.append(row)
    C = pd.DataFrame(conf); C.to_csv(RESULTS / "metrics_conflicting_evidence.csv", index=False)

    # ----------------------------- E5 stakeholder-profile sensitivity (multi-fault case)
    cases = {
      "A_safety_override": dict(type="M", air_temp_k=300.0, process_temp_k=308.9, rpm=1420, torque_nm=47.0, tool_wear_min=205,
                                observations="S_COOLANT_LOW;S_STALL;S_NOISE"),
      "B_profile_dependent": dict(type="L", air_temp_k=300.0, process_temp_k=308.2, rpm=1300, torque_nm=46.0, tool_wear_min=190,
                                  observations="S_NOISE"),
    }
    prof_rows = []
    for cname, case in cases.items():
        d0 = ENG["KBS-full"].diagnose(case)
        for p in KB_FULL.profiles:
            plan = recommend(KB_FULL, d0, p)
            for rank, it in enumerate(plan["items"], 1):
                prof_rows.append(dict(case=cname, profile=p, rank=rank, fault=it["fault"], belief=it["belief"], level=it["level"],
                                      mode=it["mode"], priority=it["priority"], safety_override=it["safety_override"],
                                      machine_status=plan["status"], immediate_downtime_h=plan["immediate_downtime_h"]))
    P = pd.DataFrame(prof_rows); P.to_csv(RESULTS / "profile_sensitivity.csv", index=False)

    # --------------------------------------------------------------- E6 timing
    t0 = time.perf_counter(); kbs_scores(ENG["KBS-full"], te); kbs_ms = (time.perf_counter() - t0) / len(te) * 1000
    X = make_features(te); pipe, feats = MODELS[("RF-eng", "hdf")]
    t0 = time.perf_counter(); pipe.predict_proba(X[feats]); rf_ms = (time.perf_counter() - t0) / len(te) * 1000
    res["ms_per_case"] = {"KBS-full": kbs_ms, "RF-eng (one target)": rf_ms}
    res["macro_f1_clean"] = {m: macro_f1(te, S, m) for m in METHODS}
    res["macro_f1_clean_no_twf"] = {m: macro_f1(te, S, m, ["hdf", "pwf", "osf"]) for m in METHODS}
    res["profile_cases"] = cases
    (RESULTS / "metrics.json").write_text(json.dumps(res, indent=2))

    # ---------------------------------------------------------------- figures
    colours = {"KBS-sensor": "#2e75b6", "KBS-full": "#c00000", "DT-raw": "#a6a6a6", "RF-raw": "#7f7f7f",
               "DT-eng": "#f4b183", "RF-eng": "#ed7d31", "LR-eng": "#70ad47"}
    fm = pd.read_csv(RESULTS / "cleaning_report.json".replace("json", "json")) if False else None
    full = pd.read_csv(DATA_PROC / "ai4i2020_clean.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cnt = {k: int(full[k.lower()].sum()) for k in FAULTS}
    ax.bar(cnt.keys(), cnt.values(), color=["#2e75b6", "#c00000", "#ed7d31", "#70ad47", "#7f7f7f"])
    for i, v in enumerate(cnt.values()): ax.text(i, v + 2, str(v), ha="center", fontsize=9)
    ax.set_ylabel("records"); ax.set_title(f"Failure modes in the dataset (failure rate {full.machine_failure.mean():.2%})", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "fig_failure_counts.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2)); w = 0.115; labs = list(TARGETS.values())
    for j, m in enumerate(METHODS):
        vals = [T1[(T1.method == m) & (T1.target == l)].f1.iloc[0] for l in labs]
        ax.bar(np.arange(len(labs)) + (j - 3) * w, vals, w, label=m, color=colours[m])
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs); ax.set_ylabel("F1 score (test, clean sensors)"); ax.set_ylim(0, 1.08)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1)); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig_f1_comparison.png", dpi=170); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, kind, xl in zip(axs, ["noise", "missing"], ["Sensor noise (x baseline sigma)", "Fraction of sensor values missing"]):
        for m in METHODS:
            s = R[(R.experiment == kind) & (R.method == m)]
            ax.plot(s.level, s.macro_f1, marker="o", ms=4, label=m, color=colours[m], lw=2.2 if m == "KBS-full" else 1.3)
        ax.set_xlabel(xl); ax.grid(alpha=0.3)
    axs[0].set_ylabel("macro-F1 over TWF/HDF/PWF/OSF"); axs[1].legend(fontsize=7.5, ncol=2)
    fig.tight_layout(); fig.savefig(FIGS / "fig_robustness.png", dpi=170); plt.close(fig)

    pd.set_option("display.width", 200)
    print(T1.round(3).to_string(index=False)); print("\ntop1:", {k: round(v, 3) for k, v in top1.items()})
    print("cost:", cost); print("\nRobustness\n", R.pivot_table(index=["experiment", "level"], columns="method", values="macro_f1").round(3).to_string())
    print("\nNoTWF\n", R.pivot_table(index=["experiment","level"], columns="method", values="macro_f1_no_twf").round(3).to_string()); print("\nConflicts\n", C.to_string(index=False)); print("\nProfiles\n", P.to_string(index=False)); print("\nms/case", res["ms_per_case"])

if __name__ == "__main__":
    main()
