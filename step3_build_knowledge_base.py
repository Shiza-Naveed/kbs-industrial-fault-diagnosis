"""
STEP 3 - Construct the knowledge base (knowledge representation).

Formalism (hybrid, justified in the report):
  * Ontology / semantic network  -> equipment, components, symptoms, faults, causes, actions
                                    stored as (subject, relation, object) triples
  * Frames                       -> slot-filler descriptions of components, symptoms, faults
  * Production rules + certainty factors -> diagnostic reasoning (rules.json)
  * Decision table               -> maintenance actions with safety / downtime / cost
  * Stakeholder profiles         -> weights used to prioritise actions
Output files are written to knowledge_base/ (knowledge is separated from the engine).
"""
import json, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kbs.config import *

MATZKA = "Matzka (2020), AI4I 2020 dataset documentation"
AUTHOR = "Author assumption - illustrative expert estimate (validate with maintenance engineer)"

# ----------------------------------------------------------------- frames
SYMPTOMS = {
    "S_NOISE":        "Unusual grinding / squealing noise during cutting",
    "S_VIB":          "Excessive vibration or chatter",
    "S_FINISH":       "Poor surface finish on the machined part",
    "S_HOT":          "Spindle housing or coolant hot to the touch",
    "S_COOLANT_LOW":  "Low coolant flow / coolant alarm",
    "S_BURN":         "Electrical burning smell",
    "S_TRIP":         "Drive overload trip or lights dimming (voltage dip)",
    "S_STALL":        "Spindle straining or stalling under load",
}
FAULTS = {
    "TWF": dict(name="Tool Wear Failure", component="Cutting_Tool",
                detect="tool_wear_min near/above 200 min (failure at a random point in 200-240 min)",
                causes={"C_ABRASIVE_WEAR": "Normal abrasive wear at end of tool life",
                        "C_EXCESS_CUT_TIME": "Excess cumulative cutting time without tool change",
                        "C_WRONG_TOOL_GRADE": "Tool grade unsuitable for the workpiece material"},
                symptoms=["S_NOISE", "S_FINISH", "S_VIB"]),
    "HDF": dict(name="Heat Dissipation Failure", component="Cooling_System",
                detect="(process - air temperature) < 8.6 K AND rotational speed < 1380 rpm",
                causes={"C_COOLANT_LOW": "Low coolant level or flow",
                        "C_FILTER_BLOCK": "Blocked coolant filter or failing pump",
                        "C_LOW_AIRFLOW": "Poor heat removal at low spindle speed"},
                symptoms=["S_HOT", "S_COOLANT_LOW"]),
    "PWF": dict(name="Power Failure", component="Motor_Drive",
                detect="power = torque x omega < 3500 W or > 9000 W",
                causes={"C_SUPPLY_VOLT": "Supply voltage fluctuation",
                        "C_DRIVE_FAULT": "Drive / motor winding degradation",
                        "C_SETPOINT": "Wrong torque / speed set-point"},
                symptoms=["S_BURN", "S_TRIP"]),
    "OSF": dict(name="Overstrain Failure", component="Spindle_Assembly",
                detect="tool_wear x torque > 11000 / 12000 / 13000 min*Nm for type L / M / H",
                causes={"C_EXCESS_DOC": "Excessive depth of cut or feed rate",
                        "C_DULL_TOOL_LOAD": "Worn tool raising the cutting load",
                        "C_BEARING": "Spindle bearing degradation"},
                symptoms=["S_STALL", "S_VIB"]),
    "RNF": dict(name="Random Failure", component="Machine",
                detect="none - occurs with ~0.1 % probability irrespective of parameters",
                causes={"C_UNKNOWN": "Unknown transient event"},
                symptoms=[]),
}
COMPONENTS = {
    "Cutting_Tool":     dict(monitored_by="tool_wear_min", description="Milling cutter / insert"),
    "Cooling_System":   dict(monitored_by="delta_t (process - air temperature)", description="Coolant pump, filter, fans"),
    "Motor_Drive":      dict(monitored_by="power_w (torque x speed)", description="Motor, VFD, power supply"),
    "Spindle_Assembly": dict(monitored_by="strain (tool wear x torque), rpm", description="Spindle, bearings, tool holder"),
}

# ------------------------------------------------------------------ actions
ACTIONS = [
 # id, fault, description, safety_risk(1-5 if ignored), downtime_h, cost_usd, spare parts, source
 ("A_TWF", "TWF", "Replace cutting tool/insert and re-zero tool offset", 2, 0.5, 60, "Insert / tool", AUTHOR),
 ("A_HDF", "HDF", "Stop spindle, let it cool, inspect coolant level, filter and pump; clean/refill", 3, 2.0, 150, "Coolant, filter", AUTHOR),
 ("A_PWF", "PWF", "Isolate power, inspect drive, motor windings and supply voltage; verify torque set-point", 5, 4.0, 400, "Fuses, drive parts", AUTHOR),
 ("A_OSF", "OSF", "Reduce feed/depth of cut; inspect spindle bearings, tool holder and clamping", 4, 3.0, 300, "Bearings, holder", AUTHOR),
 ("A_RNF", "RNF", "Operator inspection, log the event and monitor for recurrence", 1, 0.25, 20, "None", AUTHOR),
]
PROFILES = {
  "safety_first":     dict(w_safety=0.60, w_availability=0.20, w_cost=0.20, description="Safety officer: minimise risk to people and equipment"),
  "balanced":         dict(w_safety=0.40, w_availability=0.30, w_cost=0.30, description="Default plant policy"),
  "production_first": dict(w_safety=0.30, w_availability=0.50, w_cost=0.20, description="Production manager: protect availability and throughput"),
  "cost_first":       dict(w_safety=0.30, w_availability=0.20, w_cost=0.50, description="Maintenance budget owner: minimise repair cost"),
}
STAKEHOLDERS = {
  "Machine_Operator":   ["Early, understandable warnings", "Clear next action", "Ability to report observations"],
  "Maintenance_Engineer": ["Correct fault + probable cause", "Spare parts / skills needed", "Explanation of how conclusion was reached"],
  "Safety_Officer":     ["No unsafe operation", "Hard stop on high-severity faults"],
  "Production_Manager": ["Minimal unplanned downtime", "Production continuity"],
  "Maintenance_Budget_Owner": ["Low repair and downtime cost", "Avoid unnecessary part replacement"],
}

# -------------------------------------------------------------------- rules
def C(f, op, v): return {"fact": f, "op": op, "value": v}
def OBS(code): return {"fact": "obs", "op": "has", "value": code}
RULES = [
 # --- sensor / threshold rules (CF later calibrated on historical records) ---
 dict(id="R01", fault="HDF", kind="sensor", cf=0.95, overrides=["R02"], conditions=[C("delta_t","<",8.6), C("rpm","<",1380)],
      text="IF temperature gap < 8.6 K AND speed < 1380 rpm THEN heat dissipation failure", source=MATZKA),
 dict(id="R01b", fault="HDF", kind="sensor", cf=0.50, overrides=["R02"], conditions=[C("delta_t",">=",8.6), C("delta_t","<=",8.6), C("rpm","<",1380)],
      text="IF temperature gap is exactly at the 8.6 K limit AND speed < 1380 rpm THEN heat dissipation failure possible (boundary zone)", source="Author finding: labels at the 8.6 K limit depend on floating-point rounding in the dataset; CF calibrated from history"),
 dict(id="R02", fault="HDF", kind="sensor", cf=0.30, overrides=[], conditions=[C("delta_t","<",9.3), C("rpm","<",1500)],
      text="IF temperature gap < 9.3 K AND speed < 1500 rpm THEN early-warning of heat dissipation failure", source=AUTHOR),
 dict(id="R03", fault="PWF", kind="sensor", cf=0.95, overrides=["R05"], conditions=[C("power_w","<",3500)],
      text="IF power < 3500 W THEN power failure (under-power)", source=MATZKA),
 dict(id="R04", fault="PWF", kind="sensor", cf=0.95, overrides=["R06"], conditions=[C("power_w",">",9000)],
      text="IF power > 9000 W THEN power failure (over-power)", source=MATZKA),
 dict(id="R05", fault="PWF", kind="sensor", cf=0.30, overrides=[], conditions=[C("power_w","<",4000)],
      text="IF power < 4000 W THEN early-warning of under-power", source=AUTHOR),
 dict(id="R06", fault="PWF", kind="sensor", cf=0.30, overrides=[], conditions=[C("power_w",">",8500)],
      text="IF power > 8500 W THEN early-warning of over-power", source=AUTHOR),
 dict(id="R07", fault="OSF", kind="sensor", cf=0.95, overrides=["R08"], conditions=[C("strain_ratio",">",1.0)],
      text="IF wear x torque exceeds the limit for the product type THEN overstrain failure", source=MATZKA),
 dict(id="R08", fault="OSF", kind="sensor", cf=0.30, overrides=[], conditions=[C("strain_ratio",">",0.85)],
      text="IF strain is above 85 % of the limit THEN early-warning of overstrain", source=AUTHOR),
 dict(id="R09", fault="TWF", kind="sensor", cf=0.80, overrides=["R10","R11"], conditions=[C("tool_wear_min",">=",240)],
      text="IF tool wear >= 240 min THEN tool wear failure (beyond maximum life window)", source=MATZKA),
 dict(id="R10", fault="TWF", kind="sensor", cf=0.50, overrides=["R11"], conditions=[C("tool_wear_min",">=",200)],
      text="IF tool wear >= 200 min THEN tool wear failure likely (inside failure window)", source=MATZKA),
 dict(id="R11", fault="TWF", kind="sensor", cf=0.15, overrides=[], conditions=[C("tool_wear_min",">=",180)],
      text="IF tool wear >= 180 min THEN approaching end of tool life", source=AUTHOR),
 # --- operator-observation rules (weak evidence on their own) ---
 dict(id="R13", fault="TWF", kind="observation", cf=0.30, overrides=[], conditions=[OBS("S_NOISE")],
      text="IF operator reports unusual noise THEN tool wear failure (weak)", source=AUTHOR),
 dict(id="R14", fault="TWF", kind="observation", cf=0.25, overrides=[], conditions=[OBS("S_FINISH")],
      text="IF poor surface finish THEN tool wear failure (weak)", source=AUTHOR),
 dict(id="R15", fault="TWF", kind="observation", cf=0.15, overrides=[], conditions=[OBS("S_VIB")],
      text="IF excessive vibration THEN tool wear failure (very weak)", source=AUTHOR),
 dict(id="R16", fault="OSF", kind="observation", cf=0.15, overrides=[], conditions=[OBS("S_VIB")],
      text="IF excessive vibration THEN overstrain failure (very weak)", source=AUTHOR),
 dict(id="R17", fault="OSF", kind="observation", cf=0.40, overrides=[], conditions=[OBS("S_STALL")],
      text="IF spindle straining/stalling THEN overstrain failure", source=AUTHOR),
 dict(id="R18", fault="HDF", kind="observation", cf=0.35, overrides=[], conditions=[OBS("S_HOT")],
      text="IF housing/coolant hot THEN heat dissipation failure (weak)", source=AUTHOR),
 dict(id="R19", fault="HDF", kind="observation", cf=0.40, overrides=[], conditions=[OBS("S_COOLANT_LOW")],
      text="IF coolant flow low / alarm THEN heat dissipation failure", source=AUTHOR),
 dict(id="R20", fault="PWF", kind="observation", cf=0.40, overrides=[], conditions=[OBS("S_BURN")],
      text="IF electrical burning smell THEN power failure", source=AUTHOR),
 dict(id="R21", fault="PWF", kind="observation", cf=0.35, overrides=[], conditions=[OBS("S_TRIP")],
      text="IF drive overload trip / voltage dip THEN power failure (weak)", source=AUTHOR),
 # --- combined sensor + observation rules (more specific -> override weak ones) ---
 dict(id="R22", fault="TWF", kind="combined", cf=0.60, overrides=["R13"], conditions=[OBS("S_NOISE"), C("tool_wear_min",">=",150)],
      text="IF unusual noise AND tool wear >= 150 min THEN tool wear failure", source=AUTHOR),
 dict(id="R23", fault="HDF", kind="combined", cf=0.60, overrides=["R18"], conditions=[OBS("S_HOT"), C("delta_t","<",9.3)],
      text="IF hot housing AND temperature gap < 9.3 K THEN heat dissipation failure", source=AUTHOR),
 dict(id="R24", fault="HDF", kind="combined", cf=0.65, overrides=["R19"], conditions=[OBS("S_COOLANT_LOW"), C("delta_t","<",9.3)],
      text="IF coolant low AND temperature gap < 9.3 K THEN heat dissipation failure", source=AUTHOR),
 dict(id="R25", fault="PWF", kind="combined", cf=0.65, overrides=["R20"], conditions=[OBS("S_BURN"), C("power_w",">",8500)],
      text="IF burning smell AND power > 8500 W THEN power failure", source=AUTHOR),
 dict(id="R26", fault="PWF", kind="combined", cf=0.60, overrides=["R21"], conditions=[OBS("S_TRIP"), C("power_w",">",8500)],
      text="IF overload trip AND power > 8500 W THEN power failure", source=AUTHOR),
 dict(id="R27", fault="OSF", kind="combined", cf=0.65, overrides=["R17"], conditions=[OBS("S_STALL"), C("strain_ratio",">",0.8)],
      text="IF spindle straining AND strain > 80 % of limit THEN overstrain failure", source=AUTHOR),
 # --- disconfirming rules: sensor evidence that contradicts operator suspicion (negative CF) ---
 dict(id="R28", fault="HDF", kind="disconfirming", cf=-0.60, overrides=[], conditions=[C("delta_t",">=",9.8)],
      text="IF temperature gap >= 9.8 K THEN evidence AGAINST heat dissipation failure", source=AUTHOR),
 dict(id="R29", fault="PWF", kind="disconfirming", cf=-0.60, overrides=[], conditions=[C("power_w",">=",4500), C("power_w","<=",8000)],
      text="IF power within 4500-8000 W THEN evidence AGAINST power failure", source=AUTHOR),
 dict(id="R30", fault="OSF", kind="disconfirming", cf=-0.50, overrides=[], conditions=[C("strain_ratio","<",0.6)],
      text="IF strain < 60 % of limit THEN evidence AGAINST overstrain failure", source=AUTHOR),
 dict(id="R31", fault="TWF", kind="disconfirming", cf=-0.50, overrides=[], conditions=[C("tool_wear_min","<",120)],
      text="IF tool wear < 120 min THEN evidence AGAINST tool wear failure (fresh tool)", source=AUTHOR),
]

# --------------------------------------------------------------- triples
def build_triples():
    T = []
    add = lambda s, r, o, src="": T.append((s, r, o, src))
    add("CNC_Milling_Machine", "isA", "Equipment", MATZKA)
    for t, lim in STRAIN_LIMIT.items():
        add("CNC_Milling_Machine", "hasProductVariant", f"Type_{t}", MATZKA)
        add(f"Type_{t}", "hasStrainLimit", str(int(lim)), MATZKA)
    for c, d in COMPONENTS.items():
        add("CNC_Milling_Machine", "hasComponent", c, AUTHOR)
        add(c, "monitoredBy", d["monitored_by"], MATZKA)
    for f, d in FAULTS.items():
        add(f, "isA", "Fault", MATZKA); add(f, "hasName", d["name"], MATZKA)
        if d["component"] != "Machine": add(f, "affects", d["component"], AUTHOR)
        add(f, "detectedWhen", d["detect"], MATZKA)
        for cid, ctext in d["causes"].items():
            add(cid, "isA", "Cause", AUTHOR); add(f, "causedBy", cid, AUTHOR)
        for s in d["symptoms"]: add(s, "indicates", f, AUTHOR)
    for s, txt in SYMPTOMS.items(): add(s, "isA", "Symptom", AUTHOR)
    for a in ACTIONS:
        add(a[0], "isA", "MaintenanceAction", AUTHOR); add(a[1], "requires", a[0], AUTHOR)
        add(a[0], "hasSafetyRisk", str(a[3]), AUTHOR); add(a[0], "hasDowntimeHours", str(a[4]), AUTHOR); add(a[0], "hasCostUSD", str(a[5]), AUTHOR)
    for st, needs in STAKEHOLDERS.items():
        for n in needs: add(st, "concernedWith", n, AUTHOR)
    return T

def draw_graph(path):
    """Swim-lane view of the knowledge graph: Symptom -> Fault -> Causes / Action (one row per fault)."""
    from matplotlib.patches import FancyBboxPatch
    import textwrap
    fig, ax = plt.subplots(figsize=(14, 8.6)); ax.set_xlim(0, 14); ax.set_ylim(0, 9.2); ax.axis("off")
    heads = [(1.75, "Symptoms (operator)"), (5.2, "Fault  (affects component)"), (8.8, "Possible causes"), (12.35, "Maintenance action")]
    for x, t in heads: ax.text(x, 8.85, t, ha="center", fontsize=11, fontweight="bold")
    def box(x, y, w, h, txt, fc, fs=7.6, bold=False, tc="black"):
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec="#333", lw=0.8))
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)
    def arrow(x0, x1, y): ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color="#555", lw=1.3))
    acts = {a[1]: a for a in ACTIONS}
    ys = [7.5, 5.85, 4.2, 2.55, 0.9]
    for y, (f, d) in zip(ys, FAULTS.items()):
        sym = "\n".join("- " + textwrap.shorten(SYMPTOMS[s], 60, placeholder="...") for s in d["symptoms"]) or "(no reliable symptom)"
        box(1.75, y, 3.3, 1.35, sym, "#fbe5d6", fs=6.6)
        comp = d["component"].replace("_", " ")
        box(5.2, y, 3.0, 1.35, f"{f}\n{d['name']}\n({comp})", "#c00000", fs=8.5, bold=True, tc="white")
        cs = "\n".join("- " + textwrap.shorten(c, 60, placeholder="...") for c in d["causes"].values())
        box(8.8, y, 3.5, 1.35, cs, "#deebf7", fs=6.8)
        a = acts[f]
        box(12.35, y, 3.1, 1.35, "\n".join(textwrap.wrap(a[2], 36)) + f"\n[safety {a[3]}/5 | {a[4]} h | ${a[5]}]", "#e2f0d9", fs=7.0)
        arrow(3.42, 3.68, y); arrow(6.72, 7.03, y); arrow(10.58, 10.78, y)
    ax.text(3.55, 8.35, "indicates", fontsize=7.5, ha="center", style="italic")
    ax.text(6.88, 8.35, "causedBy", fontsize=7.5, ha="center", style="italic")
    ax.text(10.68, 8.35, "requires", fontsize=7.5, ha="center", style="italic")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def main():
    triples = build_triples()
    with open(KB_DIR / "ontology_triples.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject", "relation", "object", "source"]); w.writerows(triples)
    frames = {
        "equipment": {"name": "CNC_Milling_Machine", "components": COMPONENTS, "product_variants": {t: {"strain_limit": v} for t, v in STRAIN_LIMIT.items()}},
        "symptoms": {k: {"description": v} for k, v in SYMPTOMS.items()},
        "faults": {k: {kk: vv for kk, vv in v.items()} for k, v in FAULTS.items()},
    }
    (KB_DIR / "equipment_frames.json").write_text(json.dumps(frames, indent=2))
    (KB_DIR / "rules.json").write_text(json.dumps(RULES, indent=2))
    (KB_DIR / "stakeholder_profiles.json").write_text(json.dumps({"profiles": PROFILES, "stakeholders": STAKEHOLDERS}, indent=2))
    with open(KB_DIR / "maintenance_actions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["action_id", "fault", "action", "safety_risk_1to5", "downtime_h", "cost_usd", "spare_parts", "source"])
        w.writerows(ACTIONS)
    sources = [
     ["AI4I 2020 Predictive Maintenance Dataset", "Tabular sensor data + failure labels (historical records)", "10,000 rows x 14 cols", "Sensor evidence, failure labels, rule calibration and evaluation", "Matzka (2020), doi:10.1109/AI4I49448.2020.00023; Kaggle: stephanmatzka/predictive-maintenance-dataset-ai4i-2020; UCI id 601"],
     ["Failure-mode physics (thresholds)", "Documented expert knowledge", "5 failure modes", "Rules R01, R03, R04, R07, R09, R10", "Matzka (2020) dataset documentation (UCI / Kaggle)"],
     ["Maintenance actions, causes, safety/downtime/cost estimates", "Author-built expert knowledge file", "5 actions, 13 causes", "Maintenance recommendation and prioritisation", "Author assumption (illustrative); structure inspired by fault-mode / cause / strategy chain of Men et al., IEEE DataPort, doi:10.21227/7y2m-ap98"],
     ["Simulated operator observations", "Synthetic observations (incomplete, conflicting)", "10,000 rows (~5 % carry a report)", "Test incomplete / conflicting evidence handling", "Generated by step4_enrich_dataset.py (seeded)"],
    ]
    with open(KB_DIR / "knowledge_sources.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["source", "type", "size", "purpose", "citation"]); w.writerows(sources)
    draw_graph(FIGS / "fig_ontology_graph.png")
    print(f"triples={len(triples)} rules={len(RULES)} actions={len(ACTIONS)} -> {KB_DIR}")

if __name__ == "__main__":
    main()
