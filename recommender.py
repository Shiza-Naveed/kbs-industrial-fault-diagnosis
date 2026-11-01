"""
Maintenance recommendation + stakeholder-aware prioritisation.

priority = belief x ( w_safety*S + w_availability*A + w_cost*C )
   S = safety risk of the fault / 5,   A = downtime / max downtime,   C = repair cost / max cost
The weights come from the selected stakeholder profile (safety_first, balanced, production_first, cost_first).

Hard safety constraint (cannot be traded off by any profile):
   a fault with safety risk >= 4 that is PROBABLE or CONFIRMED  ->  IMMEDIATE action.
"""
from .config import *
from .engine import level_of

def recommend(kb, diag, profile="balanced"):
    w = kb.profiles[profile]
    acts = kb.actions
    max_dt = max(float(a["downtime_h"]) for a in acts.values())
    max_cost = max(float(a["cost_usd"]) for a in acts.values())
    items = []
    for fault, b in diag.belief.items():
        lvl = level_of(b)
        if lvl == "NONE": continue
        a = acts[fault]
        S, A, C = int(a["safety_risk_1to5"]) / 5, float(a["downtime_h"]) / max_dt, float(a["cost_usd"]) / max_cost
        score = b * (w["w_safety"] * S + w["w_availability"] * A + w["w_cost"] * C)
        safety_override = int(a["safety_risk_1to5"]) >= 4 and lvl in ("CONFIRMED", "PROBABLE")
        if lvl == "CONFIRMED":
            mode = "IMMEDIATE" if (safety_override or profile in ("safety_first", "balanced")) else "PLANNED"
        elif lvl == "PROBABLE":
            mode = "IMMEDIATE" if safety_override else "PLANNED"
        else:
            mode = "PLANNED" if int(a["safety_risk_1to5"]) >= 4 else "MONITOR"
        items.append(dict(fault=fault, belief=b, level=lvl, action_id=a["action_id"], action=a["action"],
                          mode=mode, priority=round(score, 4), safety_override=safety_override,
                          components=dict(S=S, A=A, C=C), downtime_h=float(a["downtime_h"]), cost_usd=float(a["cost_usd"]),
                          safety_risk=int(a["safety_risk_1to5"]), spare_parts=a["spare_parts"],
                          causes=list(kb.frames["faults"][fault]["causes"].values())))
    items.sort(key=lambda x: -x["priority"])
    plan = {"profile": profile, "weights": w, "items": items,
            "status": "NORMAL" if not items else ("STOP" if any(i["mode"] == "IMMEDIATE" for i in items) else "OPERATE_WITH_PLAN"),
            "immediate_downtime_h": sum(i["downtime_h"] for i in items if i["mode"] == "IMMEDIATE"),
            "immediate_cost_usd": sum(i["cost_usd"] for i in items if i["mode"] == "IMMEDIATE")}
    return plan
