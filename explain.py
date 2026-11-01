"""Explanation facility: HOW (how was the conclusion reached) and WHY-NOT / WHAT-IF-MISSING."""
from .engine import level_of

NAMES = {"delta_t": "temperature gap (process - air)", "power_w": "mechanical power", "rpm": "rotational speed",
         "tool_wear_min": "tool wear", "strain_ratio": "strain / limit", "torque_nm": "torque"}
UNITS = {"delta_t": "K", "power_w": "W", "rpm": "rpm", "tool_wear_min": "min", "torque_nm": "Nm", "strain_ratio": ""}

def _fmt(v):
    return "n/a (missing)" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

def explain(kb, diag, plan, backward=None):
    L = []
    f = diag.facts
    L.append("EVIDENCE")
    for k in ["delta_t", "rpm", "power_w", "tool_wear_min", "strain_ratio"]:
        v = f.get(k); L.append(f"  - {NAMES[k]:32s}: {_fmt(v)} {UNITS[k] if v is not None else ''}")
    L.append(f"  - operator observations          : {', '.join(f['observations']) if f['observations'] else 'none reported'}")
    L.append("")
    L.append("HOW THE DIAGNOSIS WAS REACHED (forward chaining)")
    if not diag.fired: L.append("  No rule fired -> no fault evidence.")
    for r in diag.fired:
        L.append(f"  [{r['id']}] {r['text']}  (CF {r['cf']:+.2f}, {r['kind']}) -> {r['fault']}")
    for r, by in diag.suppressed:
        L.append(f"  [{r['id']}] suppressed: overridden by more specific rule {by}")
    L.append("")
    L.append("BELIEF PER FAULT (certainty-factor combination)")
    for fault, b in sorted(diag.belief.items(), key=lambda x: -x[1]):
        steps = diag.cf_steps[fault]
        if not steps: continue
        chain = " , ".join(f"{rid}({cf:+.2f})" for rid, cf, _ in steps)
        L.append(f"  {fault}: positive {diag.pos[fault]:.2f}, negative {diag.neg[fault]:+.2f} -> belief {b:+.2f} [{level_of(b)}]   rules: {chain}")
        if diag.neg[fault] < 0 and diag.pos[fault] > 0:
            L.append(f"        conflict: sensor evidence AGAINST {fault} reduced the operator-based belief")
    if all(not s for s in diag.cf_steps.values()): pass
    if diag.undetermined:
        L.append("")
        L.append("INCOMPLETE INFORMATION")
        shown = {}
        for r, miss in diag.undetermined:
            if miss: shown.setdefault(r["fault"], set()).update(miss)
        for fault, miss in shown.items():
            L.append(f"  - {fault} rules could not be evaluated; missing: {', '.join(sorted(miss))}")
    L.append("")
    L.append(f"RECOMMENDATION (stakeholder profile: {plan['profile']})")
    if plan["status"] == "NORMAL": L.append("  Machine status NORMAL - no maintenance action required.")
    for i, it in enumerate(plan["items"], 1):
        L.append(f"  {i}. {it['fault']} [{it['level']}, belief {it['belief']:.2f}] -> {it['mode']}: {it['action']}")
        L.append(f"     priority {it['priority']:.3f} = belief x ({plan['weights']['w_safety']:.2f}*safety {it['components']['S']:.2f} + "
                 f"{plan['weights']['w_availability']:.2f}*downtime {it['components']['A']:.2f} + {plan['weights']['w_cost']:.2f}*cost {it['components']['C']:.2f})")
        L.append(f"     est. downtime {it['downtime_h']} h, cost ${it['cost_usd']:.0f}, spare parts: {it['spare_parts']}, safety risk {it['safety_risk']}/5")
        L.append(f"     probable causes to check: {'; '.join(it['causes'])}")
        if it["safety_override"]: L.append("     SAFETY CONSTRAINT: severity >= 4/5 -> action cannot be deferred by any stakeholder profile")
    if plan["status"] != "NORMAL":
        L.append(f"  Machine status: {plan['status']} | downtime if all IMMEDIATE actions are done: {plan['immediate_downtime_h']} h, cost ${plan['immediate_cost_usd']:.0f}")
    if backward:
        L.append("")
        L.append(f"GOAL-DRIVEN CHECK (backward chaining) for hypothesis '{backward['goal']}'")
        for it in backward["rules"]:
            r = it["rule"]
            conds = "; ".join(f"{(c['fact'] if c['fact']!='obs' else 'observation '+c['value'])} {c['op'] if c['fact']!='obs' else ''} {c['value'] if c['fact']!='obs' else ''} -> {'?' if v is None else v} (value {_fmt(s) if not isinstance(s,list) else s})" for c, v, s in it["conditions"])
            tag = it["status"].upper() + (f" (overridden by {it['overridden_by']})" if it.get("overridden_by") else "")
            L.append(f"  [{r['id']}] {tag}: {conds}")
        L.append(f"  supported belief now: {backward['supported']:.2f}; best case if unknowns confirm: {backward['best_case']:.2f}")
        if backward["needs"]: L.append(f"  to settle the hypothesis, measure/ask: {', '.join(backward['needs'])}")
    return "\n".join(L)
