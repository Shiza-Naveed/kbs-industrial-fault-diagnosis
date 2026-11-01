"""
Web UI for the Knowledge-Based System:   pip install flask   then   python app.py   ->  http://127.0.0.1:5000
All reasoning is done by the same engine as the CLI (kbs/engine.py); the page only collects evidence and displays the result.
"""
import math, pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from kbs.config import *
from kbs.engine import KnowledgeBase, InferenceEngine, level_of
from kbs.recommender import recommend
from kbs.explain import explain
import demo
from kbs.query import QueryEngine, EXAMPLES
from kbs import learning

app = Flask(__name__, static_folder="ui")
kb = eng = qe = None
def load_all():
    """(Re)load the knowledge base, using certainty factors learned from technician feedback if they exist."""
    global kb, eng, qe
    kb = KnowledgeBase(learned=True); eng = InferenceEngine(kb); qe = QueryEngine(kb)
load_all()
FAULT_INFO = {f: {"name": d["name"], "component": d["component"].replace("_", " ")} for f, d in kb.frames["faults"].items()}
clean = lambda o: {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in o.items()}

@app.get("/")
def index(): return send_from_directory("ui", "index.html")

@app.get("/api/meta")
def meta():
    return jsonify(symptoms={k: v["description"] for k, v in kb.frames["symptoms"].items()}, faults=FAULT_INFO,
                   profiles={k: v["description"] for k, v in kb.profiles.items()},
                   examples=EXAMPLES, scenarios={k: {"title": t, "evidence": ev} for k, (t, ev) in demo.SCENARIOS.items()})

@app.post("/api/ask")
def ask():
    b = request.get_json(force=True)
    return jsonify(qe.ask(b.get("q", ""), b.get("context")))

@app.post("/api/feedback")
def feedback():
    """Technician feedback: which faults were really present for this evidence."""
    b = request.get_json(force=True)
    ev = {k: b.get(k) for k in ["type", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]}; ev["observations"] = ";".join(b.get("observations", []))
    n = learning.record(ev, eng.diagnose(ev), [f for f in b.get("confirmed", []) if f in FAULTS])
    return jsonify(feedback_cases=n)

@app.post("/api/relearn")
def relearn():
    changes = learning.relearn(); load_all()
    return jsonify({**learning.status(), "changes": changes})

@app.post("/api/learning/reset")
def learning_reset():
    learning.reset(); load_all(); return jsonify(**learning.status())

@app.get("/api/learning")
def learning_status(): return jsonify(**learning.status())

@app.get("/api/testrow")
def testrow():
    df = pd.read_csv(DATA_PROC / "ai4i2020_kbs_enriched.csv"); te = df[df.split == "test"].reset_index(drop=True)
    only_fail = request.args.get("failure") == "1"
    if only_fail: te = te[te.machine_failure == 1].reset_index(drop=True)
    r = te.sample(1).iloc[0]
    ev = {k: (None if pd.isna(r[k]) else r[k].item() if hasattr(r[k], "item") else r[k]) for k in ["type", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]}
    ev["observations"] = "" if pd.isna(r["observations"]) else r["observations"]
    return jsonify(evidence=ev, truth={f: int(r[f.lower()]) for f in FAULTS}, machine_failure=int(r["machine_failure"]), unexplained=int(r["unexplained_failure"]))

@app.post("/api/diagnose")
def diagnose():
    b = request.get_json(force=True); profile = b.get("profile", "balanced")
    ev = {k: b.get(k) for k in ["type", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]}
    ev["observations"] = ";".join(b.get("observations", []))
    d = eng.diagnose(ev); plan = recommend(kb, d, profile)
    f = d.facts
    return jsonify(
        facts={k: f.get(k) for k in ["delta_t", "power_w", "strain_ratio", "rpm", "tool_wear_min"]},
        beliefs=[{"fault": k, "belief": v, "level": level_of(v), "pos": d.pos[k], "neg": d.neg[k], **FAULT_INFO[k]} for k, v in d.belief.items()],
        fired=[{"id": r["id"], "text": r["text"], "cf": r["cf"], "kind": r["kind"], "fault": r["fault"]} for r in d.fired],
        suppressed=[{"id": r["id"], "by": by, "fault": r["fault"]} for r, by in d.suppressed],
        undetermined=[{"id": r["id"], "fault": r["fault"], "missing": m} for r, m in d.undetermined if m],
        plan={k: plan[k] for k in ["status", "profile", "weights", "immediate_downtime_h", "immediate_cost_usd"]},
        items=plan["items"], explanation=explain(kb, d, plan),
        backward={g: {k: v for k, v in eng.prove(g, ev).items() if k != "rules"} | {"rules": [
            {"id": it["rule"]["id"], "text": it["rule"]["text"], "status": it["status"], "overridden_by": it["overridden_by"]} for it in eng.prove(g, ev)["rules"]]}
            for g in ["TWF", "HDF", "PWF", "OSF"]})

if __name__ == "__main__":
    app.run(debug=False, port=5000)
