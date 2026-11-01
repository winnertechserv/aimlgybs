"""
Chapter 31, Example 3: Drift Detection Triggering Retraining
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This is the monitoring stage in code. It computes the Population Stability Index and a
Kolmogorov-Smirnov test across four months of production data, tracks the model's
accuracy decay against its deployment baseline, and then applies trigger logic combining
drift, performance drop and elapsed time to decide automatically when retraining should
fire.

Run it:           python ex3_drift_detection_triggering_retraining.py
Expected output:  expected_output/ex3_drift_detection_triggering_retraining.txt
"""
# Chapter 31 - Example 3: Data/model drift detection triggering retraining
import numpy as np
from scipy import stats
np.random.seed(3)

print("=== Monitoring: Detecting Drift and Triggering Retraining ===\n")
print("A deployed model silently decays when the world changes. Monitoring")
print("must detect (a) INPUT DATA DRIFT and (b) MODEL PERFORMANCE DROP.\n")

# ---------- Population Stability Index ----------
def psi(expected, actual, bins=10):
    """PSI < 0.1 no shift | 0.1-0.25 moderate | > 0.25 major shift"""
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(expected, edges)[0] / len(expected)
    a_pct = np.histogram(actual,   edges)[0] / len(actual)
    # an empty bin would give log(0) or a division by zero, so floor every share at 1e-6
    e_pct = np.clip(e_pct, 1e-6, None); a_pct = np.clip(a_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

n = 2000
reference = np.random.normal(50, 10, n)      # training-time feature distribution

months = {
  "Month 1 (same as training)"  : np.random.normal(50.0, 10.0, n),
  "Month 2 (slight shift)"      : np.random.normal(51.5, 10.0, n),
  "Month 3 (moderate shift)"    : np.random.normal(54.0, 11.0, n),
  "Month 4 (major shift)"       : np.random.normal(60.0, 14.0, n),
}

print("--- Input Data Drift: PSI and Kolmogorov-Smirnov test ---\n")
print(f"{'Production window':<30}{'PSI':>8}{'KS stat':>10}{'p-value':>11}  {'Verdict'}")
print("-" * 76)
drift_flags = {}
for name, actual in months.items():
    p = psi(reference, actual)
    ks, pv = stats.ks_2samp(reference, actual)
    if p < 0.10:   verdict = "no significant drift"
    elif p < 0.25: verdict = "MODERATE drift - watch"
    else:          verdict = "MAJOR drift - retrain"
    drift_flags[name] = p
    print(f"  {name:<28}{p:>8.4f}{ks:>10.4f}{pv:>11.2e}  {verdict}")

# ---------- Model performance drift ----------
print("\n--- Model Performance Drift ---\n")
baseline_acc = 0.912
observed = {"Month 1": 0.909, "Month 2": 0.901, "Month 3": 0.874, "Month 4": 0.812}
threshold = 0.05
print(f"Baseline test accuracy at deployment: {baseline_acc}")
print(f"Alert threshold: relative drop > {threshold*100:.0f}%\n")
print(f"{'Window':<12}{'Accuracy':>10}{'Drop':>10}  {'Action'}")
print("-" * 56)
for m, acc in observed.items():
    drop = (baseline_acc - acc) / baseline_acc
    action = "TRIGGER RETRAINING" if drop > threshold else "continue monitoring"
    print(f"  {m:<10}{acc:>10.3f}{drop*100:>9.1f}%  {action}")

# ---------- Retraining decision logic ----------
print("\n--- Automated Retraining Trigger Logic ---\n")
def should_retrain(psi_value, acc_drop, days_since_train):
    reasons = []
    if psi_value > 0.25:      reasons.append(f"data drift PSI={psi_value:.3f}")
    if acc_drop > 0.05:       reasons.append(f"accuracy drop {acc_drop*100:.1f}%")
    if days_since_train > 30: reasons.append(f"scheduled ({days_since_train} days)")
    return (len(reasons) > 0), reasons

cases = [
    ("Month 1", 0.02, 0.003, 7),
    ("Month 3", 0.18, 0.042, 21),
    ("Month 4", 0.41, 0.110, 35),
]
for label, p, drop, days in cases:
    go, why = should_retrain(p, drop, days)
    status = "RETRAIN" if go else "no action"
    print(f"  {label}: PSI={p:.2f}, drop={drop*100:.1f}%, {days}d since training -> {status}")
    for r in why:
        print(f"      reason: {r}")

print("\n=== Retraining triggers ===")
print("  1. Scheduled intervals   - e.g. retail forecasting retrains weekly")
print("  2. Data / model drift    - ride-sharing model after a big city event")
print("  3. Performance drop      - fraud detection when errors spike")
print("\nTools: Prometheus (infrastructure), Evidently AI (data drift & ML")
print("metrics), CloudWatch (cloud apps); Airflow / Kubeflow run the pipeline.")
