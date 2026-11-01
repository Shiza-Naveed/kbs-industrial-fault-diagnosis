"""STEP 8 - Draw the system architecture figure used in the report."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import json
from kbs.config import *
N_RULES = len(json.loads((KB_DIR / 'rules.json').read_text()))

fig, ax = plt.subplots(figsize=(13, 7.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 7.4); ax.axis("off")
def box(x, y, w, h, title, body="", fc="#ddebf7", tc="black", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color=tc)
    if body: ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=fs, linespacing=1.35, color=tc)
def arr(x0, y0, x1, y1, txt="", dy=0.12):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.4))
    if txt: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, txt, ha="center", fontsize=7.3, style="italic")

box(0.1, 4.6, 2.8, 2.5, "1  Knowledge sources", "AI4I 2020 sensor data\n(historical records)\nExpert knowledge file\n(actions, causes, costs)\nOperator observations\n(simulated, incomplete)", "#fff2cc")
box(3.4, 5.35, 2.5, 1.75, "2  Acquisition & learning", "validate / clean data\nfeature engineering\ncalibrate rule CFs\nrelearn CFs from\ntechnician feedback", "#fbe5d6")
box(3.4, 3.2, 2.5, 1.85, "3  Knowledge Base", f"ontology triples (114)\nframes, {N_RULES} rules + CF\naction table\nstakeholder profiles", "#e2f0d9")
box(6.4, 3.2, 3.0, 3.9, "4  Inference engine", "Working memory\n(current readings + derived facts)\n\nForward chaining (data-driven)\nBackward chaining (goal-driven)\nCertainty-factor combination\nConflict resolution:\n override > specificity > |CF| > safety\nMissing evidence -> 'undetermined'", "#c6e0b4")
box(9.9, 4.6, 3.0, 2.5, "5  Recommender", "belief x stakeholder-weighted\n(safety, downtime, cost)\nHard safety constraint\nIMMEDIATE / PLANNED / MONITOR", "#bdd7ee")
box(9.9, 2.2, 3.0, 2.0, "6  Explanation facility", "HOW: rules fired + CF steps\nWHY-NOT: refuted rules\nWHAT-IF: evidence to measure", "#d9d2e9")
box(6.4, 0.5, 3.0, 2.0, "7  User interface", "Web UI (app.py) + CLI (demo.py)\nreadings + observations in,\nlive diagnosis + explanation out", "#fce4d6")
box(0.1, 0.5, 5.8, 2.1, "8  ML baselines (comparison with alternatives)", "Decision Tree / Random Forest / Logistic Regression\ntrained on the train split; compared with the KBS on accuracy,\nrobustness (noise, missing data), explainability, cost, safety", "#ededed")
arr(2.9, 6.1, 3.4, 6.1); arr(4.65, 5.35, 4.65, 5.05, "builds"); arr(5.9, 4.1, 6.4, 4.5, "rules, frames")
arr(9.4, 5.6, 9.9, 5.6, "beliefs"); arr(11.4, 4.6, 11.4, 4.2, "trace"); arr(7.9, 2.5, 7.9, 3.2, "evidence")
arr(9.9, 3.0, 9.4, 1.9, "report to user"); arr(5.9, 1.5, 6.4, 3.5, "compared with")
fig.tight_layout(); fig.savefig(FIGS / "fig_architecture.png", dpi=170)
print("saved fig_architecture.png")
