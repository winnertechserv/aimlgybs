"""
Chapter 31, Example 1: Experiment Tracking (What MLflow Does)
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This builds a miniature MLflow: every run logs its parameters, metrics and an artifact
hash, then all runs are compared side by side exactly as the MLflow UI shows them. Five
model configurations are trained and the best is selected from the logs, and the train-
test gap column shows how logged metrics let you spot overfitting without rerunning
anything.

Run it:           python ex1_experiment_tracking_what_mlflow_does.py
Expected output:  expected_output/ex1_experiment_tracking_what_mlflow_does.txt
"""
# Chapter 31 - Example 1: MLflow-style experiment tracking (from scratch)
import json, time, hashlib, os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

np.random.seed(0)
print("=== Experiment Tracking (the core of MLflow) ===\n")
print("Without tracking you cannot answer: which run produced this model?")
print("what data and hyperparameters were used? can we reproduce it?\n")

STORE = "mlruns.json"   # written next to where you run the script
runs = []

def log_run(experiment, params, metrics, artifact_bytes):
    """Equivalent of mlflow.log_params / log_metrics / log_model."""
    run = {
        "run_id"    : hashlib.md5(f"{experiment}{params}".encode()).hexdigest()[:8],
        "experiment": experiment,
        "params"    : params,
        "metrics"   : metrics,
        "artifact_hash": hashlib.md5(artifact_bytes).hexdigest()[:12],
    }
    runs.append(run)
    return run

# ---- Data ----
data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
print(f"Dataset: {data.data.shape[0]} samples, {data.data.shape[1]} features\n")

# ---- Run several experiments, logging each one ----
configs = [
    ("logreg",       LogisticRegression(C=0.1,  max_iter=2000, random_state=0), {"model":"LogisticRegression","C":0.1}),
    ("logreg",       LogisticRegression(C=1.0,  max_iter=2000, random_state=0), {"model":"LogisticRegression","C":1.0}),
    ("logreg",       LogisticRegression(C=10.0, max_iter=2000, random_state=0), {"model":"LogisticRegression","C":10.0}),
    ("random_forest",RandomForestClassifier(n_estimators=50,  random_state=0),  {"model":"RandomForest","n_estimators":50}),
    ("random_forest",RandomForestClassifier(n_estimators=200, random_state=0),  {"model":"RandomForest","n_estimators":200}),
]

print("Running experiments and logging every run...\n")
for exp, model, params in configs:
    model.fit(Xtr, ytr)
    pred_tr, pred_te = model.predict(Xtr), model.predict(Xte)
    metrics = {
        "train_accuracy": round(accuracy_score(ytr, pred_tr), 4),
        "test_accuracy" : round(accuracy_score(yte, pred_te), 4),
        "test_f1"       : round(f1_score(yte, pred_te), 4),
    }
    artifact = repr(getattr(model, "coef_", getattr(model, "feature_importances_", ""))).encode()
    r = log_run(exp, params, metrics, artifact)
    print(f"  run {r['run_id']} | {exp:<14} | {str(params):<48} | test_acc={metrics['test_accuracy']}")

with open(STORE, "w") as f:
    json.dump(runs, f, indent=2)

# ---- The "MLflow UI": compare runs ----
print(f"\n=== Run Comparison (what the MLflow UI shows) ===\n")
print(f"{'run_id':<10}{'experiment':<16}{'train_acc':>10}{'test_acc':>10}{'test_f1':>9}  {'artifact'}")
print("-" * 78)
for r in runs:
    print(f"{r['run_id']:<10}{r['experiment']:<16}"
          f"{r['metrics']['train_accuracy']:>10}{r['metrics']['test_accuracy']:>10}"
          f"{r['metrics']['test_f1']:>9}  {r['artifact_hash']}")

best = max(runs, key=lambda r: r["metrics"]["test_accuracy"])
print(f"\nBest run: {best['run_id']}  ({best['params']})")
print(f"  test accuracy = {best['metrics']['test_accuracy']}")

# ---- Overfitting check straight from the logs ----
print("\n=== Generalisation check from logged metrics ===\n")
for r in runs:
    gap = r["metrics"]["train_accuracy"] - r["metrics"]["test_accuracy"]
    flag = "OVERFIT" if gap > 0.05 else "ok"
    print(f"  {r['run_id']}  train-test gap = {gap:+.4f}  [{flag}]")

print(f"\nAll runs persisted to {STORE} ({os.path.getsize(STORE)} bytes)")
print("This is exactly what MLflow does: log params, metrics and artifacts")
print("for every run so results are comparable, auditable and reproducible.")
