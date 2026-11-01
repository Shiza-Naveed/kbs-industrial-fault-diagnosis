"""
Query interface of the Knowledge-Based System: ask questions in plain English.

No external AI is used. A question is
  1. linked to knowledge-base entities (faults, symptoms, components, rules, stakeholders, profiles),
  2. mapped to an intent by keyword rules,
  3. answered by looking the facts up in the knowledge base (ontology triples, frames, rules, action table),
     or - when the text describes a machine's readings - by running the inference engine.
Every answer lists its sources (which triples / rules / files it came from).
"""
import csv, math, re
from .config import *
from .engine import KnowledgeBase, InferenceEngine, level_of
from .recommender import recommend
from .explain import explain

FAULT_SYN = {
    "TWF": ["tool wear failure", "twf", "worn tool", "worn out", "worn", "wear out", "tool life", "blunt tool", "tool wear"],
    "HDF": ["heat dissipation failure", "heat dissipation", "hdf", "overheating", "overheat", "cooling failure", "cooling system failure"],
    "PWF": ["power failure", "pwf", "electrical failure", "motor drive failure"],
    "OSF": ["overstrain failure", "overstrain", "osf", "over strain", "mechanical overload"],
    "RNF": ["random failure", "rnf"],
}
SYMP_SYN = {
    "S_NOISE": ["s_noise", "grinding", "squeal", "unusual noise", "noise"], "S_VIB": ["s_vib", "vibration", "vibrating", "chatter"],
    "S_FINISH": ["s_finish", "surface finish", "poor finish", "rough surface"], "S_HOT": ["s_hot", "hot to the touch", "hot housing", "housing is hot", "hot spindle", "overheated housing", "running hot", "feels hot"],
    "S_COOLANT_LOW": ["s_coolant_low", "coolant alarm", "low coolant", "coolant flow", "coolant low", "coolant"], "S_BURN": ["s_burn", "burning smell", "burnt smell", "burning", "smell"],
    "S_TRIP": ["s_trip", "overload trip", "tripping", "voltage dip", "lights dimming", "trips", "trip"], "S_STALL": ["s_stall", "stalling", "stalls", "stall", "straining", "strain on the spindle"],
}
COMP_SYN = {"Cutting_Tool": ["cutting tool", "insert", "milling cutter"], "Cooling_System": ["cooling system", "cooling", "coolant pump", "pump", "filter"],
            "Motor_Drive": ["motor drive", "motor", "vfd", "power supply", "drive"], "Spindle_Assembly": ["spindle assembly", "spindle", "bearing", "tool holder"]}
STAKE_SYN = {"Machine_Operator": ["operator"], "Maintenance_Engineer": ["maintenance engineer", "engineer"], "Safety_Officer": ["safety officer"],
             "Production_Manager": ["production manager"], "Maintenance_Budget_Owner": ["budget owner", "budget"]}
PROFILES = ["safety_first", "balanced", "production_first", "cost_first"]
SENSOR_NAME = {"air_temp_k": "air temperature", "process_temp_k": "process temperature", "rpm": "rotational speed", "torque_nm": "torque", "tool_wear_min": "tool wear", "type": "product type"}
EXAMPLES = ["What causes heat dissipation failure?", "What symptoms indicate PWF?", "How do I fix overstrain and how long does it take?", "What does a burning smell indicate?",
            "Which fault has the highest safety risk?", "Explain rule R22", "What is the threshold for HDF?",
            "Type L, air temperature 300, process temperature 308.2, rpm 1300, torque 46, tool wear 90"]

def _find(text, table):
    """Return the keys of `table` whose synonyms occur in text (longest synonym first, no overlaps)."""
    found, t = [], " " + text + " "
    for key, syns in sorted(table.items(), key=lambda kv: -max(len(s) for s in kv[1])):
        for s in sorted(syns, key=len, reverse=True):
            m = re.search(r"(?<![a-z_])" + re.escape(s) + r"s?(?![a-z_])", t)
            if m:
                found.append(key); t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]; break
    return found

def _bullets(items): return "\n".join("• " + i for i in items)

class QueryEngine:
    def __init__(self, kb: KnowledgeBase = None):
        self.kb = kb or KnowledgeBase(); self.eng = InferenceEngine(self.kb)
        with open(KB_DIR / "ontology_triples.csv") as fh: self.triples = list(csv.DictReader(fh))
        self.faults = self.kb.frames["faults"]; self.symptoms = self.kb.frames["symptoms"]; self.acts = self.kb.actions

    # ------------------------------------------------------------ helpers
    def _src(self, *parts): return [p for p in parts if p]
    def _fname(self, f): return f"{self.faults[f]['name']} ({f})"
    def _triples(self, s=None, r=None, o=None):
        return [t for t in self.triples if (s is None or t["subject"] == s) and (r is None or t["relation"] == r) and (o is None or t["object"] == o)]
    def _tsrc(self, ts): return [f"ontology: ({t['subject']}, {t['relation']}, {t['object']})" for t in ts[:6]]

    def parse_evidence(self, text):
        t = text.lower(); ev = {}
        def num(pats):
            for p in pats:
                m = re.search(p, t)
                if m: return float(m.group(1))
        ev["rpm"] = num([r"(\d{3,4}(?:\.\d+)?)\s*rpm", r"(?:rpm|speed)\D{0,10}?(\d{3,4}(?:\.\d+)?)"])
        ev["air_temp_k"] = num([r"air\s*temp\w*\D{0,12}?(\d{3}(?:\.\d+)?)"])
        ev["process_temp_k"] = num([r"process\s*temp\w*\D{0,12}?(\d{3}(?:\.\d+)?)"])
        ev["torque_nm"] = num([r"torque\D{0,12}?(\d+(?:\.\d+)?)", r"(\d+(?:\.\d+)?)\s*nm\b"])
        ev["tool_wear_min"] = num([r"wear\D{0,12}?(\d+(?:\.\d+)?)", r"(\d+(?:\.\d+)?)\s*min(?:ute)?s?\b"])
        power = num([r"(\d{3,5}(?:\.\d+)?)\s*(?:w|watts?)\b", r"power\D{0,12}?(\d{3,5}(?:\.\d+)?)"])
        derived = None
        if power is not None and ev["torque_nm"] is None and ev["rpm"]:
            ev["torque_nm"] = round(power * 60 / (2 * math.pi * ev["rpm"]), 1); derived = f"torque {ev['torque_nm']} Nm derived from {power:.0f} W at {ev['rpm']:.0f} rpm"
        m = re.search(r"\b(?:type|quality)\s*([lmh])\b", t) or re.search(r"\b(low|medium|high)\s*(?:quality|type)", t)
        ev["type"] = ({"low": "L", "medium": "M", "high": "H"}.get(m.group(1), m.group(1).upper()) if m else None)
        ev["observations"] = ";".join(_find(t, SYMP_SYN))
        numeric = any(ev[k] is not None for k in SENSOR_NAME if k != "type") or power is not None
        return ev, numeric, derived

    # -------------------------------------------------------------- intent
    def ask(self, text, context=None):
        context = dict(context or {}); q = re.sub(r"\s+", " ", text.strip()); t = q.lower()
        if not q: return self._help("Ask me something about the knowledge base.")
        ev, numeric, derived = self.parse_evidence(q)
        faults = _find(t, FAULT_SYN); symps = _find(t, SYMP_SYN); comps = _find(t, COMP_SYN)
        m = re.search(r"\brule\s*(r\d+[a-z]?)\b|\b(r\d{2}[a-z]?)\b", t)
        word = lambda *ws: any(re.search(r"\b" + w, t) for w in ws)
        fault = faults[0] if faults else context.get("fault")
        used_ctx = not faults and bool(context.get("fault"))

        if word("help", "what can you do", "what can i ask") and not faults: return self._help()
        if numeric or word("diagnose", "diagnosis"): return self._diagnose(ev, derived, q)
        if m: return self._rule_detail((m.group(1) or m.group(2)).upper())
        if not faults and word("highest", "most", "lowest", "least", "cheapest", "fastest", "longest", "shortest", "rank", "biggest", "smallest") and word("safety", "risk", "cost", "downtime", "dangerous", "expensive", "cheap", "fast", "long", "short", "repair", "time"):
            return self._rank(t)
        if word("who") and word("safety", "production", "budget", "operator", "engineer", "responsible", "cares"): return self._stake(t)
        if word("stakeholder", "who cares", "profile", "weights", "priorit") and not word("fault"): return self._stake(t)
        if symps and not faults and (len(symps) > 1 or word("wrong", "problem", "issue", "what is it")) and not word("mean", "indicate", "suggest"):
            return self._diagnose(ev, derived, q)
        if symps and word("mean", "indicate", "suggest", "sign of", "what fault", "which fault", "point to", "imply", "if i see", "caused by"): return self._from_symptom(symps[0])
        if symps and not faults and word("report", "see", "hear", "smell", "notice", "observ", "seeing", "hearing"):
            return self._diagnose(ev, derived, q)
        if not comps and not faults and word("component", "components", "parts of the machine"): return self._list_components()
        if comps and not faults and word("fault", "failure", "fail", "affect", "monitor", "component", "go wrong", "problem"): return self._component(comps[0])
        if word("rules", "rule") and fault: return self._rules_for(fault)
        if word("threshold", "limit", "when does", "when do", "trigger", "detect", "criteria", "how is", "how do you know", "boundary") and fault: return self._threshold(fault)
        if word("cause", "reason", "why does", "why do", "root", "originat") and fault: return self._causes(fault, used_ctx)
        if word("symptom", "sign", "recogni", "notice", "look for", "indicator", "indicate", "how can i tell", "how do i know", "how to tell", "tell if") and fault: return self._symptoms(fault, used_ctx)
        attr = [a for a, ws in {"downtime": ("downtime", "how long", "time", "hours", "duration"), "cost": ("cost", "how much", "price", "expens", "budget", "dollar"),
                                "spare": ("spare", "part", "parts", "need to replace", "replace"), "safety": ("safe", "risk", "danger", "severe", "hazard")}.items() if word(*ws)]
        if fault and attr: return self._attributes(fault, attr, used_ctx)
        if word("action", "fix", "repair", "maintenance", "what should", "what do i do", "how to", "how do i", "recommend", "solve", "resolve") and fault: return self._action(fault, used_ctx)
        if word("list", "all", "which", "what") and word("fault", "failure", "failures", "faults", "modes") and not fault: return self._list_faults()
        if word("list", "all", "which", "what") and word("symptom", "symptoms", "observation"): return self._list_symptoms()
        if fault: return self._overview(fault)
        if symps: return self._from_symptom(symps[0])
        if comps: return self._component(comps[0])
        return self._help("I could not map that question to something in the knowledge base.", t)

    # ------------------------------------------------------------- answers
    def _ans(self, intent, text, sources=(), context=None, **extra):
        return {"intent": intent, "answer": text, "sources": list(sources), "context": context or {}, "followups": extra.pop("followups", []), **extra}

    def _overview(self, f):
        a = self.acts[f]; fr = self.faults[f]
        txt = (f"**{self._fname(f)}** affects the {fr['component'].replace('_', ' ').lower()}.\n• Detection: {fr['detect']}\n• Causes: {'; '.join(fr['causes'].values())}\n"
               f"• Typical symptoms: {', '.join(self.symptoms[s]['description'] for s in fr['symptoms']) or 'none reliable'}\n• Action: {a['action']} (safety {a['safety_risk_1to5']}/5, {a['downtime_h']} h, ${a['cost_usd']})")
        return self._ans("overview", txt, self._tsrc(self._triples(f)) + ["actions: maintenance_actions.csv"], {"fault": f},
                         followups=[f"What causes {f}?", f"What symptoms indicate {f}?", f"How do I fix {f}?", f"Which rules conclude {f}?"])

    def _causes(self, f, ctx):
        ts = self._triples(f, "causedBy"); fr = self.faults[f]
        txt = f"**Possible causes of {self._fname(f)}**" + (" (continuing about " + f + ")" if ctx else "") + "\n" + _bullets(fr["causes"].values())
        return self._ans("causes", txt + f"\n\nWhen it is detected: {fr['detect']}.", self._tsrc(ts), {"fault": f}, followups=[f"How do I fix {f}?", f"What symptoms indicate {f}?"])

    def _symptoms(self, f, ctx):
        fr = self.faults[f]
        if not fr["symptoms"]:
            return self._ans("symptoms", f"**{self._fname(f)}** has no reliable symptom: it occurs at random, irrespective of the operating conditions, so neither sensors nor operators can anticipate it.", [], {"fault": f})
        ts = self._triples(None, "indicates", f)
        rules = [r for r in self.kb.rules if r["fault"] == f and r["kind"] == "observation"]
        det = "\n".join(f"• {self.symptoms[r['conditions'][0]['value']]['description']} -> CF {r['cf']:.2f} on its own" for r in rules)
        txt = (f"**Symptoms that indicate {self._fname(f)}**\n" + _bullets(self.symptoms[s]["description"] + f" ({s})" for s in fr["symptoms"]) +
               (f"\n\nHow much each report is trusted alone (calibrated certainty factor):\n{det}" if det else "") + "\nA symptom alone stays at watch level; it is much stronger when a sensor corroborates it.\nSensor criterion: " + fr["detect"] + ".")
        return self._ans("symptoms", txt, self._tsrc(ts) + [f"rules {', '.join(r['id'] for r in rules)}"], {"fault": f}, followups=[f"What causes {f}?", f"How do I fix {f}?"])

    def _action(self, f, ctx):
        a = self.acts[f]
        txt = (f"**Recommended action for {self._fname(f)}**\n{a['action']}\n• Estimated downtime: {a['downtime_h']} h\n• Estimated cost: ${a['cost_usd']}\n• Spare parts: {a['spare_parts']}\n• Safety risk if ignored: {a['safety_risk_1to5']}/5" +
               ("\n• **Safety constraint:** severity >= 4, so a probable or confirmed fault is always acted on immediately, whatever the stakeholder profile." if int(a["safety_risk_1to5"]) >= 4 else "") +
               "\n(Cost, downtime and safety values are expert estimates, flagged as assumptions in the report.)")
        return self._ans("action", txt, self._tsrc(self._triples(f, "requires")) + ["actions: maintenance_actions.csv"], {"fault": f}, followups=[f"What causes {f}?", f"What is the threshold for {f}?"])

    def _attributes(self, f, attr, ctx):
        a = self.acts[f]; L = []
        if "downtime" in attr: L.append(f"Downtime: **{a['downtime_h']} h**")
        if "cost" in attr: L.append(f"Cost: **${a['cost_usd']}**")
        if "spare" in attr: L.append(f"Spare parts: **{a['spare_parts']}**")
        if "safety" in attr: L.append(f"Safety risk if ignored: **{a['safety_risk_1to5']}/5**")
        return self._ans("attributes", f"**{self._fname(f)}** - action: {a['action']}\n" + _bullets(L), ["actions: maintenance_actions.csv (author estimates)"], {"fault": f}, followups=[f"How do I fix {f}?", "Which fault has the highest safety risk?"])

    def _threshold(self, f):
        fr = self.faults[f]; rules = [r for r in self.kb.rules if r["fault"] == f and r["kind"] == "sensor"]
        txt = f"**How {self._fname(f)} is detected:** {fr['detect']}.\n\nSensor rules (calibrated certainty factor):\n" + _bullets(f"[{r['id']}] {r['text']} (CF {r['cf']:.2f})" for r in rules)
        if f == "HDF": txt += "\n\nNote: labels at exactly 8.6 K depend on floating-point rounding in the dataset, so rule R01b treats that boundary zone as only 'possible'."
        return self._ans("threshold", txt if rules else txt + "\nNo sensor rule exists for this fault.", [f"rules {', '.join(r['id'] for r in rules)}", "frames: equipment_frames.json"], {"fault": f})

    def _rules_for(self, f):
        rules = [r for r in self.kb.rules if r["fault"] == f]
        if not rules: return self._ans("rules", f"No rule concludes {f}; it cannot be diagnosed from evidence.", [], {"fault": f})
        txt = f"**Rules that conclude {self._fname(f)}** ({len(rules)})\n" + _bullets(f"[{r['id']}] {r['kind']}, CF {r['cf']:+.2f}: {r['text']}" for r in rules)
        return self._ans("rules", txt, [f"rules_calibrated.json: {', '.join(r['id'] for r in rules)}"], {"fault": f}, followups=[f"Explain rule {rules[0]['id']}"])

    def _rule_detail(self, rid):
        r = self.kb.rule_by_id.get(rid) or self.kb.rule_by_id.get(rid[0] + rid[1:].lower())
        if not r: return self._help(f"There is no rule {rid}. Rules are R01 ... R31 (plus R01b).")
        ov = r.get("overrides") or []
        txt = (f"**Rule {r['id']}** ({r['kind']}) concludes {self._fname(r['fault'])}\n{r['text']}\n• Certainty factor: {r['cf']:+.2f}" +
               (f" (expert prior {r['prior_cf']:+.2f}, calibrated on {r.get('support', '?')} training records, precision {r.get('precision_train')})" if "prior_cf" in r else " (expert value)") +
               (f"\n• Overrides: {', '.join(ov)}" if ov else "") + (f"\n• Learned from {r['feedback_n']} technician feedback cases (CF was {r['learned_from_cf']:+.2f})" if "feedback_n" in r else "") + f"\n• Source: {r['source']}")
        return self._ans("rule", txt, [f"rules_calibrated.json: {r['id']}"], {"fault": r["fault"]}, followups=[f"Which rules conclude {r['fault']}?"])

    def _from_symptom(self, s):
        fs = [t["object"] for t in self._triples(s, "indicates")]
        rules = [r for r in self.kb.rules if r["kind"] in ("observation", "combined") and any(c["fact"] == "obs" and c["value"] == s for c in r["conditions"])]
        txt = f"**{self.symptoms[s]['description']}** ({s}) points to: " + ", ".join(self._fname(f) for f in fs) + ".\n" + _bullets(f"[{r['id']}] {r['text']} (CF {r['cf']:.2f})" for r in rules)
        txt += "\n\nOn its own a symptom is weak evidence (watch level); a sensor reading that agrees makes it strong, and one that contradicts it removes the suspicion."
        return self._ans("symptom", txt, self._tsrc(self._triples(s, "indicates")) + [f"rules {', '.join(r['id'] for r in rules)}"], {"fault": fs[0]} if fs else {}, followups=[f"What causes {fs[0]}?"] if fs else [])

    def _component(self, c):
        fs = [t["subject"] for t in self._triples(None, "affects", c)]
        mon = (self._triples(c, "monitoredBy") or [{"object": "-"}])[0]["object"]
        txt = f"**{c.replace('_', ' ')}** is monitored by: {mon}.\nFaults that affect it: " + (", ".join(self._fname(f) for f in fs) or "none") + "."
        return self._ans("component", txt, self._tsrc(self._triples(None, "affects", c)) + self._tsrc(self._triples(c, "monitoredBy")), {"fault": fs[0]} if fs else {})

    def _rank(self, t):
        key, rev, lab = (("safety_risk_1to5", True, "safety risk") if re.search(r"safe|risk|danger", t) else ("cost_usd", True, "repair cost") if re.search(r"cost|expens|cheap|price", t)
                         else ("downtime_h", True, "downtime"))
        if re.search(r"lowest|least|cheap|fastest|shortest|smallest", t): rev = False
        rows = sorted(self.acts.items(), key=lambda kv: float(kv[1][key]), reverse=rev)
        txt = f"**Faults ranked by {lab} ({'highest' if rev else 'lowest'} first)**\n" + _bullets(f"{f} - {self.faults[f]['name']}: {a[key]}" for f, a in rows)
        return self._ans("rank", txt, ["actions: maintenance_actions.csv"], {"fault": rows[0][0]}, followups=[f"How do I fix {rows[0][0]}?"])

    def _stake(self, t):
        if re.search(r"profile|weight|priorit", t):
            txt = "**Stakeholder profiles (weights for safety / availability / cost)**\n" + _bullets(f"{k}: {v['w_safety']:.2f} / {v['w_availability']:.2f} / {v['w_cost']:.2f} - {v['description']}" for k, v in self.kb.profiles.items())
            return self._ans("profiles", txt + "\nA hard safety constraint (severity >= 4) overrides every profile.", ["stakeholder_profiles.json"])
        picked = _find(t, STAKE_SYN)
        items = {k: v for k, v in self.kb.stakeholders.items() if not picked or k in picked}
        return self._ans("stakeholders", "**Stakeholders and what they need**\n" + "\n".join(f"{k.replace('_', ' ')}:\n" + _bullets(v) for k, v in items.items()), self._tsrc(self._triples(None, "concernedWith")))

    def _list_faults(self):
        txt = "**Failure modes in the knowledge base**\n" + _bullets(f"{f} - {d['name']} (affects {d['component'].replace('_', ' ').lower()})" for f, d in self.faults.items())
        return self._ans("faults", txt, self._tsrc(self._triples(None, "isA", "Fault")), followups=["Which fault has the highest safety risk?"])

    def _list_components(self):
        cs = self.kb.frames["equipment"]["components"]
        return self._ans("components", "**Components of the CNC milling machine**\n" + _bullets(f"{c.replace('_', ' ')} - {d['description']} (monitored by {d['monitored_by']})" for c, d in cs.items()), self._tsrc(self._triples("CNC_Milling_Machine", "hasComponent")))

    def _list_symptoms(self):
        return self._ans("symptoms_list", "**Operator observations the system understands**\n" + _bullets(f"{d['description']} ({k})" for k, d in self.symptoms.items()), self._tsrc(self._triples(None, "isA", "Symptom")))

    def _help(self, lead="I answer questions from the knowledge base and can diagnose a machine from a description.", t=""):
        guess = ""
        return self._ans("help", f"{lead}\n\n**Try asking:**\n" + _bullets(EXAMPLES), [], followups=EXAMPLES[:4])

    def _diagnose(self, ev, derived, q):
        full = {k: ev.get(k) for k in ["type", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]}; full["observations"] = ev.get("observations", "")
        d = self.eng.diagnose(full); plan = recommend(self.kb, d, "balanced")
        got = [f"{SENSOR_NAME[k]} {v}" for k, v in full.items() if v is not None and k != "observations"] + ([f"observations: {full['observations'].replace(';', ', ')}"] if full["observations"] else [])
        miss = [SENSOR_NAME[k] for k in SENSOR_NAME if full.get(k) is None]
        head = {"NORMAL": "Machine NORMAL - no fault evidence above the watch level.", "OPERATE_WITH_PLAN": "OPERATE WITH A MAINTENANCE PLAN - no immediate stop required.", "STOP": "STOP - immediate action required."}[plan["status"]]
        lines = [f"**Understood:** {', '.join(got) or 'no readings'}" + (f" ({derived})" if derived else ""), f"**Result:** {head}"]
        for it in plan["items"]:
            lines.append(f"• {it['fault']} - {self.faults[it['fault']]['name']}: **{it['level']}** (belief {it['belief']:.2f}) -> **{it['mode']}**: {it['action']}")
        if miss: lines.append(f"Not given (treated as unknown, never guessed): {', '.join(miss)}.")
        if plan["items"]:
            bw = self.eng.prove(plan["items"][0]["fault"], full)
            if bw["needs"]: lines.append(f"Measuring {', '.join(bw['needs'])} could raise the {plan['items'][0]['fault']} belief to as much as {bw['best_case']:.2f}.")
        top = plan["items"][0]["fault"] if plan["items"] else None
        return self._ans("diagnosis", "\n".join(lines), [f"rules fired: {', '.join(r['id'] for r in d.fired) or 'none'}", "inference engine: forward + backward chaining"], {"fault": top} if top else {},
                         evidence=full, explanation=explain(self.kb, d, plan), followups=["Why? (see the explanation)"] + ([f"What causes {top}?"] if top else []))
