"""
Learning component - command line.
    python learn.py --status     number of stored technician feedback cases and learned changes
    python learn.py --relearn    update certainty factors from the feedback log (writes knowledge_base/rules_learned.json)
    python learn.py --reset      forget all feedback and learned values (original knowledge is untouched)
Feedback is recorded from the web UI (python app.py): after a diagnosis choose "Correct" or "Wrong" and name the real fault.
"""
import argparse, json
from kbs import learning

ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--status", action="store_true"); g.add_argument("--relearn", action="store_true"); g.add_argument("--reset", action="store_true")
a = ap.parse_args()
if a.relearn:
    ch = learning.relearn()
    print(f"Updated {len(ch)} rules from {learning.count()} feedback cases:")
    for c in ch: print(f"  {c['rule']:5} {c['fault']}  CF {c['old_cf']:+.2f} -> {c['new_cf']:+.2f}   ({c['feedback_cases']} cases, {c['feedback_confirmed']} confirmed)")
elif a.reset: learning.reset(); print("Feedback log and learned certainty factors removed.")
else: print(json.dumps(learning.status(), indent=2))
