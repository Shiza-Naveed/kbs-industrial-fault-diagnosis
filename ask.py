"""
Ask the Knowledge-Based System questions in plain English.
    python ask.py                      interactive chat (type 'quit' to leave)
    python ask.py "What causes HDF?"   single question
Follow-up questions work ("What about the cost?" refers to the fault just discussed).
"""
import sys
from kbs.query import QueryEngine, EXAMPLES

def show(r):
    print("\n" + r["answer"])
    if r["sources"]: print("\n  [sources] " + " | ".join(r["sources"][:4]))
    if r["followups"]: print("  [you could ask] " + "  /  ".join(r["followups"][:3]))
    print()

def main():
    qe = QueryEngine(); ctx = {}
    if len(sys.argv) > 1: show(qe.ask(" ".join(sys.argv[1:]))); return
    print("Knowledge-Based System - ask me about faults, causes, symptoms, actions, rules, costs, or describe a machine's readings.")
    print("Examples:\n  " + "\n  ".join(EXAMPLES[:5]) + "\n(type 'quit' to leave)\n")
    while True:
        try: text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if text.lower() in ("quit", "exit", "q"): break
        r = qe.ask(text, ctx); ctx = r.get("context") or ctx; show(r)

if __name__ == "__main__":
    main()
