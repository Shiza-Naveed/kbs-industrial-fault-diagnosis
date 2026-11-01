"""Tests for the question-answering layer (python tests/test_query.py)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kbs.query import QueryEngine
q = QueryEngine()
def A(text, ctx=None): return q.ask(text, ctx)

def test_causes():
    r = A("What causes heat dissipation failure?"); assert r["intent"] == "causes" and "coolant" in r["answer"].lower() and r["sources"]
def test_symptoms():
    r = A("What symptoms indicate PWF?"); assert r["intent"] == "symptoms" and "burning" in r["answer"].lower()
def test_action_and_attributes():
    r = A("How do I fix overstrain?"); assert r["intent"] == "action" and "bearing" in r["answer"].lower() and "Safety constraint" in r["answer"]
    r = A("How long does it take to repair HDF and how much does it cost?"); assert "2.0 h" in r["answer"] and "$150" in r["answer"]
def test_symptom_to_fault():
    r = A("What does a burning smell indicate?"); assert r["intent"] == "symptom" and "PWF" in r["answer"]
def test_rank():
    r = A("Which fault has the highest safety risk?"); assert r["intent"] == "rank" and r["answer"].split("\n")[1].startswith("• PWF")
    r = A("Which repair is the cheapest?"); assert r["answer"].split("\n")[1].startswith("• RNF")
def test_rule_detail():
    r = A("Explain rule R22"); assert r["intent"] == "rule" and "noise" in r["answer"].lower() and "Certainty factor" in r["answer"]
    assert "R01b" in A("Which rules conclude HDF?")["answer"]
def test_threshold():
    r = A("What is the threshold for HDF?"); assert r["intent"] == "threshold" and "8.6" in r["answer"]
def test_context_followup():
    r1 = A("What causes tool wear failure?"); r2 = A("What about the cost?", r1["context"])
    assert r2["intent"] == "attributes" and "$60" in r2["answer"]
def test_diagnose_from_text():
    r = A("Type L, air temperature 300, process temperature 308.2, rpm 1300, torque 46, tool wear 90")
    assert r["intent"] == "diagnosis" and "HDF" in r["answer"] and "CONFIRMED" in r["answer"] and r["evidence"]["rpm"] == 1300
def test_power_converted_and_missing_reported():
    r = A("power 9500 W at 1500 rpm, type M"); assert "PWF" in r["answer"] and "derived" in r["answer"] and "Not given" in r["answer"]
def test_operator_report_conflict():
    r = A("rpm 1520 torque 41, air temperature 300, process temperature 310, wear 60, operator smells burning"); assert "NORMAL" in r["answer"]
def test_lists_and_stakeholders():
    assert "RNF" in A("list all failure modes")["answer"]; assert "safety_first" in A("show the stakeholder profiles")["answer"]
def test_fallback_helps():
    r = A("what is the weather tomorrow"); assert r["intent"] == "help" and r["followups"]

def test_regressions_from_probing():
    assert A("is overstrain dangerous?")["intent"] == "attributes" and "4/5" in A("is overstrain dangerous?")["answer"]
    assert A("what should I do about random failures")["intent"] == "action" and "RNF" in A("what should I do about random failures")["answer"]
    assert A("which components can fail")["intent"] == "components"
    assert "Safety Officer" in A("who is responsible for safety")["answer"]
    assert "PWF" not in A("what is S_STALL")["answer"] and "OSF" in A("what is S_STALL")["answer"]
    r = A("spindle is straining and vibrating, what is wrong"); assert r["intent"] == "diagnosis" and "OSF" in r["answer"] or "S_STALL" in r["answer"]
    r = A("how can I tell if the tool is worn out"); assert r["intent"] == "symptoms" and "200" in r["answer"]
    assert A("explain r01")["intent"] == "rule"

if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns: f(); print("PASS", f.__name__)
    print(f"{len(fns)} query tests passed")
