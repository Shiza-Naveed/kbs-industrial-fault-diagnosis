"""
Interactive / scripted demonstration of the KBS.

  python demo.py --all                       run 7 canned scenarios -> results/demo_cases.txt
  python demo.py --scenario hdf              run one canned scenario
  python demo.py --row 123 --profile cost_first   diagnose row 123 of the TEST split
  python demo.py --interactive               type in readings (press Enter to leave a value UNKNOWN)
Options: --profile safety_first|balanced|production_first|cost_first   --goal HDF (backward-chaining check)
"""
import argparse, sys
import pandas as pd
from kbs.config import *
from kbs.engine import KnowledgeBase, InferenceEngine
from kbs.recommender import recommend
from kbs.explain import explain

SCENARIOS = {
 "normal":   ("Normal operation", dict(type="M", air_temp_k=300.2, process_temp_k=310.1, rpm=1520, torque_nm=41.0, tool_wear_min=60, observations="")),
 "hdf":      ("Heat dissipation failure (sensor evidence + operator report)", dict(type="L", air_temp_k=300.0, process_temp_k=308.2, rpm=1300, torque_nm=46.0, tool_wear_min=90, observations="S_HOT")),
 "conflict": ("Conflicting evidence: operator smells burning, sensors say power is normal", dict(type="M", air_temp_k=299.8, process_temp_k=309.7, rpm=1540, torque_nm=40.0, tool_wear_min=80, observations="S_BURN")),
 "twf":      ("Tool near end of life + operator hears noise", dict(type="H", air_temp_k=301.0, process_temp_k=311.2, rpm=1480, torque_nm=38.0, tool_wear_min=215, observations="S_NOISE")),
 "missing":  ("Incomplete information: air-temperature sensor offline", dict(type="M", air_temp_k=None, process_temp_k=307.0, rpm=1450, torque_nm=42.0, tool_wear_min=100, observations="S_COOLANT_LOW")),
 "osf":      ("Overstrain: wear x torque above the limit for type L", dict(type="L", air_temp_k=300.4, process_temp_k=310.0, rpm=1380, torque_nm=62.0, tool_wear_min=195, observations="S_STALL")),
 "multi":    ("Multiple concurrent faults (safety constraint vs stakeholder profile)", dict(type="M", air_temp_k=300.0, process_temp_k=308.9, rpm=1420, torque_nm=47.0, tool_wear_min=205, observations="S_COOLANT_LOW;S_STALL;S_NOISE")),
}
GOAL = {"hdf": "HDF", "conflict": "PWF", "twf": "TWF", "missing": "HDF", "osf": "OSF", "multi": "OSF", "normal": "HDF"}

def run(title, ev, profile, goal, kb, eng):
    d = eng.diagnose(ev); plan = recommend(kb, d, profile)
    bw = eng.prove(goal, ev) if goal else None
    return f"{'=' * 100}\nCASE: {title}\nINPUT: {ev}\n{'=' * 100}\n" + explain(kb, d, plan, bw) + "\n"

def ask(prompt, cast=float):
    v = input(f"  {prompt} (Enter = unknown): ").strip()
    if v == "": return None
    try: return cast(v)
    except ValueError: print("  not a number -> treated as unknown"); return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--all", action="store_true"); ap.add_argument("--scenario", choices=SCENARIOS)
    ap.add_argument("--row", type=int); ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--profile", default="balanced", choices=["safety_first", "balanced", "production_first", "cost_first"])
    ap.add_argument("--goal", help="fault code for the backward-chaining check, e.g. HDF")
    a = ap.parse_args()
    kb = KnowledgeBase(); eng = InferenceEngine(kb)
    if a.all:
        out = "".join(run(t, ev, a.profile, GOAL[k], kb, eng) + "\n" for k, (t, ev) in SCENARIOS.items())
        (RESULTS / "demo_cases.txt").write_text(out); print(out)
    elif a.scenario:
        t, ev = SCENARIOS[a.scenario]; print(run(t, ev, a.profile, a.goal or GOAL[a.scenario], kb, eng))
    elif a.row is not None:
        df = pd.read_csv(DATA_PROC / "ai4i2020_kbs_enriched.csv"); r = df[df.split == "test"].reset_index(drop=True).iloc[a.row]
        ev = {k: r[k] for k in ["type", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]}
        ev["observations"] = "" if pd.isna(r["observations"]) else r["observations"]
        print(run(f"Test row {a.row}", ev, a.profile, a.goal, kb, eng))
        print("GROUND TRUTH (dataset labels):", {f: int(r[f.lower()]) for f in FAULTS}, "| machine_failure =", int(r["machine_failure"]))
    elif a.interactive:
        print("Enter the current readings. Press Enter to leave a value unknown (incomplete information).")
        ev = {"type": (input("  product type L/M/H (Enter = unknown): ").strip().upper() or None),
              "air_temp_k": ask("air temperature [K]"), "process_temp_k": ask("process temperature [K]"), "rpm": ask("rotational speed [rpm]"),
              "torque_nm": ask("torque [Nm]"), "tool_wear_min": ask("tool wear [min]")}
        print("  symptom codes:", ", ".join(kb.frames["symptoms"]))
        ev["observations"] = input("  operator observations, ';'-separated (Enter = none): ").strip()
        print(run("Interactive session", ev, a.profile, a.goal, kb, eng))
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
