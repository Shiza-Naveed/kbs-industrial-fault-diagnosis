"""
Learning component of the Knowledge-Based System: learning from technician feedback.

Offline calibration (step 5) fixes the certainty factors (CF) from historical records. This module lets the running
system keep learning: after a diagnosis a technician confirms it or names the fault that was really present. Every
feedback case is stored (append-only log); `relearn()` then updates the CF of each learnable rule with the same
Bayesian shrinkage used in calibration, now counting the feedback cases together with the historical ones:

        CF = (positives + K * prior_CF) / (support + K)      K = 10, bounded to [0.05, 0.99]

where a feedback case is a 'support' for every rule that fired in it and a 'positive' when the rule's fault was confirmed.
Safety properties: the original knowledge files are never overwritten (learned CFs go to rules_learned.json, which can be
deleted to roll back); disconfirming rules and the safety constraint are not learned; one case moves a CF only slightly.
"""
import json, time
from .config import *

LEARN_KINDS = ("sensor", "observation", "combined")
FEEDBACK_LOG = KB_DIR / "feedback_log.jsonl"
LEARNED = KB_DIR / "rules_learned.json"
K = 10

def _jsonable(v):
    try: return v.item()
    except AttributeError: return v

def record(evidence, diagnosis, confirmed_faults, path=FEEDBACK_LOG):
    """Store one technician feedback case: the evidence, the rules that fired, and the faults confirmed to be present."""
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "evidence": {k: _jsonable(v) for k, v in evidence.items()},
             "fired": [r["id"] for r in diagnosis.fired], "belief": {f: round(b, 3) for f, b in diagnosis.belief.items()},
             "confirmed": sorted(set(confirmed_faults))}
    with open(path, "a") as fh: fh.write(json.dumps(entry) + "\n")
    return count(path)

def load(path=FEEDBACK_LOG):
    return [json.loads(l) for l in open(path)] if path.exists() else []

def count(path=FEEDBACK_LOG): return len(load(path))

def relearn(base=KB_DIR / "rules_calibrated.json", feedback=FEEDBACK_LOG, out=LEARNED):
    """Recompute the CF of learnable rules from historical counts + feedback. Returns the list of CF changes."""
    rules = json.loads(base.read_text()); cases = load(feedback); changes = []
    for r in rules:
        if r["kind"] not in LEARN_KINDS: continue
        fired = [c for c in cases if r["id"] in c["fired"]]
        if not fired: continue
        n0 = int(r.get("support") or 0); p0 = round((r.get("precision_train") or 0.0) * n0)
        conf = sum(1 for c in fired if r["fault"] in c["confirmed"])
        n = n0 + len(fired); p = p0 + conf
        prior = r.get("prior_cf", r["cf"]); new = round(min(0.99, max(0.05, (p + K * prior) / (n + K))), 2)
        changes.append({"rule": r["id"], "fault": r["fault"], "kind": r["kind"], "old_cf": r["cf"], "new_cf": new,
                        "feedback_cases": len(fired), "feedback_confirmed": conf})
        r["learned_from_cf"] = r["cf"]; r["cf"] = new; r["feedback_n"] = len(fired)
    out.write_text(json.dumps(rules, indent=2))
    return changes

def reset(feedback=FEEDBACK_LOG, out=LEARNED):
    """Forget everything learned from feedback (the original calibrated knowledge is untouched)."""
    for p in (feedback, out):
        if p.exists(): p.unlink()

def status(feedback=FEEDBACK_LOG, out=LEARNED):
    changed = []
    if out.exists():
        changed = [{"rule": r["id"], "old_cf": r["learned_from_cf"], "new_cf": r["cf"], "feedback_cases": r["feedback_n"]}
                   for r in json.loads(out.read_text()) if "feedback_n" in r]
    return {"feedback_cases": count(feedback), "learned_rules_file": out.exists(), "changes": changed}
