"""Run the complete pipeline end to end:  python run_all.py
Pipeline: inject impurities into the original Kaggle file -> wrangle -> verify -> knowledge base -> enrich -> calibrate -> ML -> evaluate."""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
steps = ["step0_inject_impurities.py", "step2_clean_data.py", "step2b_verify_cleaning.py", "step3_build_knowledge_base.py", "step4_enrich_dataset.py",
         "step5_calibrate_rules.py", "step6_train_ml.py", "step7_evaluate.py", "step8_make_architecture_figure.py", "step9_learning_demo.py"]
if not (ROOT / "data/raw/ai4i2020_original.csv").exists():
    sys.exit("Put the original Kaggle file at data/raw/ai4i2020_original.csv first")
for s in steps:
    print(f"\n>>> {s}"); r = subprocess.run([sys.executable, s], cwd=ROOT)
    if r.returncode: sys.exit(f"FAILED at {s}")
print("\n>>> tests"); subprocess.run([sys.executable, "tests/test_kbs.py"], cwd=ROOT); subprocess.run([sys.executable, "tests/test_query.py"], cwd=ROOT); subprocess.run([sys.executable, "tests/test_learning.py"], cwd=ROOT)
print("\n>>> demo cases"); subprocess.run([sys.executable, "demo.py", "--all"], cwd=ROOT, stdout=subprocess.DEVNULL)
print("\nDone. Results in results/, knowledge base in knowledge_base/, figures in results/figures/.")
