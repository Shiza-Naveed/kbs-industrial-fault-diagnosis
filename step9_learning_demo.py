"""
STEP 9 - Demonstrate the learning component with SIMULATED technician feedback (a demonstration of the mechanism, not a performance claim).
Operator-only 'coolant alarm' reports (rule R19) are replayed with three feedback histories; the resulting certainty factors are saved.
"""
import json, tempfile; from pathlib import Path
from kbs.config import *; from kbs import learning
from kbs.engine import KnowledgeBase, InferenceEngine, level_of
base = KB_DIR / "rules_calibrated.json"; eng = InferenceEngine(KnowledgeBase())
EV = dict(type="M", air_temp_k=None, process_temp_k=None, rpm=1520, torque_nm=41.0, tool_wear_min=60, observations="S_COOLANT_LOW")   # temperature sensors offline: only the operator report speaks
d = eng.diagnose(EV); get = lambda p, rid: {r["id"]: r["cf"] for r in json.loads(Path(p).read_text())}[rid]
out = {"rule": "R19", "text": KnowledgeBase().rule_by_id["R19"]["text"], "calibrated_cf": get(base, "R19"), "scenarios": []}
for name, conf, rej in [("technicians confirm 40 of 40 reports", 40, 0), ("technicians confirm 20, reject 20", 20, 20), ("technicians reject 40 of 40 reports", 0, 40)]:
    with tempfile.TemporaryDirectory() as t:
        fb, lr = Path(t) / "fb.jsonl", Path(t) / "l.json"
        for _ in range(conf): learning.record(EV, d, ["HDF"], fb)
        for _ in range(rej): learning.record(EV, d, [], fb)
        learning.relearn(base, fb, lr); new = get(lr, "R19")
        kb2 = KnowledgeBase(); kb2.rules = json.loads(lr.read_text()); kb2.rule_by_id = {r["id"]: r for r in kb2.rules}
        b = InferenceEngine(kb2).diagnose(EV).belief["HDF"]
    out["scenarios"].append({"feedback": name, "new_cf": new, "hdf_belief_for_the_same_report": b, "level": level_of(b)})
out["belief_before"] = d.belief["HDF"]; out["level_before"] = level_of(d.belief["HDF"])
(RESULTS / "learning_demo.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
