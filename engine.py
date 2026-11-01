"""
Inference engine of the Knowledge-Based System.

* Forward chaining  : data-driven.  facts -> derived facts -> diagnosis rules -> fault beliefs
* Backward chaining : goal-driven.  "Is fault X present?" -> which rules / evidence prove or refute it,
                      and which missing evidence would settle the question
* Uncertainty       : MYCIN-style certainty factors (Shortliffe & Buchanan, 1975)
* Conflict resolution (agenda ordering):
      1. an overridden (less specific) rule is suppressed when its more specific rule fires
      2. more specific rules (more conditions) first
      3. higher |certainty factor| first
      4. rules for more safety-critical faults first
* Incomplete information: a rule whose facts are missing is NOT fired; it is recorded as
  'undetermined' together with the missing fact, so the explanation can say what to measure next.
"""
import json, math
from dataclasses import dataclass, field
from .config import *

OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
       ">=": lambda a, b: a >= b, "==": lambda a, b: a == b}
BASE_FACTS_FOR = {  # which raw measurements each (derived) fact needs - used for 'what to measure next'
    "delta_t": ["process_temp_k", "air_temp_k"], "power_w": ["torque_nm", "rpm"],
    "strain_ratio": ["tool_wear_min", "torque_nm", "type"], "rpm": ["rpm"], "tool_wear_min": ["tool_wear_min"]}

def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))

def combine_cf(a, b):
    """MYCIN certainty-factor combination."""
    if a >= 0 and b >= 0: return a + b * (1 - a)
    if a <= 0 and b <= 0: return a + b * (1 + a)
    return (a + b) / (1 - min(abs(a), abs(b)))

def level_of(cf):
    if cf >= LEVEL_CONFIRMED: return "CONFIRMED"
    if cf >= LEVEL_PROBABLE: return "PROBABLE"
    if cf >= LEVEL_WATCH: return "WATCH"
    return "NONE"

class KnowledgeBase:
    """Loads the knowledge from files in knowledge_base/ (knowledge is separate from the engine)."""
    def __init__(self, calibrated=True, learned=False):
        rf = KB_DIR / ("rules_calibrated.json" if calibrated and (KB_DIR / "rules_calibrated.json").exists() else "rules.json")
        if learned and (KB_DIR / "rules_learned.json").exists(): rf = KB_DIR / "rules_learned.json"   # CFs updated from technician feedback
        self.rules_file = rf.name
        self.rules = json.loads(rf.read_text())
        self.frames = json.loads((KB_DIR / "equipment_frames.json").read_text())
        prof = json.loads((KB_DIR / "stakeholder_profiles.json").read_text())
        self.profiles, self.stakeholders = prof["profiles"], prof["stakeholders"]
        import csv
        with open(KB_DIR / "maintenance_actions.csv") as fh:
            self.actions = {r["fault"]: r for r in csv.DictReader(fh)}
        self.strain_limit = {t: v["strain_limit"] for t, v in self.frames["equipment"]["product_variants"].items()}
        self.rule_by_id = {r["id"]: r for r in self.rules}
        # safety severity per fault (from the action table) -> used in conflict resolution
        self.fault_safety = {f: int(a["safety_risk_1to5"]) for f, a in self.actions.items()}

    def without(self, kinds):
        """Return a copy of the KB with some rule kinds removed (used for ablation studies)."""
        import copy
        kb = copy.copy(self); kb.rules = [r for r in self.rules if r["kind"] not in kinds]
        kb.rule_by_id = {r["id"]: r for r in kb.rules}; return kb

@dataclass
class Diagnosis:
    facts: dict
    fired: list = field(default_factory=list)        # rules that fired (after conflict resolution)
    suppressed: list = field(default_factory=list)   # (overridden rule, overridden by)
    undetermined: list = field(default_factory=list) # rules that could not be evaluated (missing facts)
    belief: dict = field(default_factory=dict)       # fault -> final certainty factor
    pos: dict = field(default_factory=dict)
    neg: dict = field(default_factory=dict)
    cf_steps: dict = field(default_factory=dict)     # fault -> list of (rule, cf, running_total)
    cycles: int = 0

    def level(self, fault): return level_of(self.belief.get(fault, 0.0))
    def faults_at_least(self, thr): return sorted([f for f, b in self.belief.items() if b >= thr], key=lambda f: -self.belief[f])

class InferenceEngine:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    # ---------------------------------------------------------------- facts
    def derive_facts(self, evidence):
        """Layer-0 forward chaining: derive engineering facts from raw measurements."""
        f = {k: (v.item() if hasattr(v, "item") else v) for k, v in evidence.items()}
        obs = f.get("observations") or []
        if isinstance(obs, str): obs = [o for o in obs.replace(",", ";").split(";") if o.strip()]
        f["observations"] = [o.strip() for o in obs]
        g = lambda k: None if _missing(f.get(k)) else f.get(k)
        f["delta_t"] = None if g("process_temp_k") is None or g("air_temp_k") is None else round(g("process_temp_k") - g("air_temp_k"), 2)
        f["power_w"] = None if g("torque_nm") is None or g("rpm") is None else g("torque_nm") * g("rpm") * 2 * math.pi / 60
        lim = self.kb.strain_limit.get(g("type"))
        f["strain_ratio"] = None if g("tool_wear_min") is None or g("torque_nm") is None or lim is None else g("tool_wear_min") * g("torque_nm") / lim
        return f

    def _cond(self, c, facts):
        """Evaluate one condition -> True / False / None(unknown)."""
        if c["fact"] == "obs":
            return bool(c["value"] in facts["observations"])
        v = facts.get(c["fact"])
        if _missing(v): return None
        return bool(OPS[c["op"]](float(v), c["value"]))

    def _rule_status(self, rule, facts):
        res = [self._cond(c, facts) for c in rule["conditions"]]
        if any(r is False for r in res): return "false", res
        if any(r is None for r in res): return "unknown", res
        return "true", res

    # ------------------------------------------------------- forward chaining
    def diagnose(self, evidence) -> Diagnosis:
        facts = self.derive_facts(evidence)
        d = Diagnosis(facts=facts, cycles=2)            # cycle 1: derive facts, cycle 2: diagnosis rules
        conflict_set = []
        for r in self.kb.rules:
            st, res = self._rule_status(r, facts)
            if st == "true": conflict_set.append(r)
            elif st == "unknown":
                miss = []
                for c, v in zip(r["conditions"], res):
                    if v is None: miss += BASE_FACTS_FOR.get(c["fact"], [c["fact"]])
                d.undetermined.append((r, sorted(set(m for m in miss if _missing(evidence.get(m))))))
        # conflict resolution step 1: overridden rules are suppressed by the specific rule that fired
        fired_ids = {r["id"] for r in conflict_set}
        keep = []
        for r in conflict_set:
            by = next((o["id"] for o in conflict_set if r["id"] in o.get("overrides", [])), None)
            if by: d.suppressed.append((r, by))
            else: keep.append(r)
        # steps 2-4: specificity, |CF|, fault safety severity
        keep.sort(key=lambda r: (-len(r["conditions"]), -abs(r["cf"]), -self.kb.fault_safety.get(r["fault"], 1), r["id"]))
        d.fired = keep
        for fault in FAULTS: d.pos[fault] = 0.0; d.neg[fault] = 0.0; d.cf_steps[fault] = []
        for r in keep:                                  # fire in agenda order
            f = r["fault"]
            if r["cf"] >= 0:
                d.pos[f] = combine_cf(d.pos[f], r["cf"]); run = d.pos[f]
            else:
                d.neg[f] = combine_cf(d.neg[f], r["cf"]); run = d.neg[f]
            d.cf_steps[f].append((r["id"], r["cf"], round(run, 3)))
        for f in FAULTS: d.belief[f] = round(combine_cf(d.pos[f], d.neg[f]), 4)
        return d

    # ------------------------------------------------------ backward chaining
    def prove(self, goal_fault, evidence):
        """Goal-driven reasoning for the hypothesis 'goal_fault is present'.
        Returns, per candidate rule, whether it is supported / refuted / undetermined,
        the evidence still needed, and the best belief reachable if the unknowns turn out favourably."""
        facts = self.derive_facts(evidence)
        out = {"goal": goal_fault, "rules": [], "needs": set(), "supported": 0.0}
        cands = []
        for r in self.kb.rules:
            if r["fault"] != goal_fault or r["cf"] < 0: continue
            st, res = self._rule_status(r, facts)
            cands.append((r, st, res))
        true_ids = {r["id"] for r, st, _ in cands if st == "true"}
        optimistic = 0.0
        for r, st, res in cands:
            by = next((o["id"] for o, s2, _ in cands if s2 == "true" and r["id"] in o.get("overrides", [])), None)
            status = {"true": "supported", "false": "refuted", "unknown": "undetermined"}[st]
            item = {"rule": r, "status": status, "overridden_by": by, "conditions": []}
            for c, v in zip(r["conditions"], res):
                shown = facts["observations"] if c["fact"] == "obs" else facts.get(c["fact"])
                item["conditions"].append((c, v, shown))
                if v is None: out["needs"].update(m for m in BASE_FACTS_FOR.get(c["fact"], [c["fact"]]) if _missing(evidence.get(m)))
            out["rules"].append(item)
            if by: continue                      # conflict resolution: overridden rule adds nothing
            if st == "true": out["supported"] = combine_cf(out["supported"], r["cf"])
            if st in ("true", "unknown"): optimistic = combine_cf(optimistic, r["cf"])
        out["needs"] = sorted(out["needs"]); out["best_case"] = round(optimistic, 3)
        out["supported"] = round(out["supported"], 3)
        return out
