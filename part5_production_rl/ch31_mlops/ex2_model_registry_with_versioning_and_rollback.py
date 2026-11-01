"""
Chapter 31, Example 2: Model Registry with Versioning and Rollback
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This builds the registry concepts described above: register successive versions with
their data hash and code commit, promote through Staging to Production, and - when v3
degrades in production - roll back to the last known-good version in one step. The audit
trail at the end is exactly what a postmortem or a regulator would ask for.

Run it:           python ex2_model_registry_with_versioning_and_rollback.py
Expected output:  expected_output/ex2_model_registry_with_versioning_and_rollback.txt
"""
# Chapter 31 - Example 2: Model registry with versioning, staging and rollback
import json, hashlib, datetime
import numpy as np

print("=== Model Registry: Versioning, Promotion and Rollback ===\n")
print("A registry answers: which model is in production RIGHT NOW, what data")
print("and code produced it, and how do we roll back if it misbehaves?\n")

class ModelRegistry:
    def __init__(self):
        self.versions = []          # all registered versions
        self.stages = {"Production": None, "Staging": None}
        self.audit = []

    def register(self, name, metrics, data_hash, code_commit):
        v = len(self.versions) + 1
        entry = {"name": name, "version": v, "metrics": metrics,
                 "data_hash": data_hash, "code_commit": code_commit,
                 "stage": "None",
                 "registered": f"2026-03-{10+v:02d}"}
        self.versions.append(entry)
        self.audit.append(f"registered v{v}")
        return entry

    def transition(self, version, stage):
        prev = self.stages.get(stage)
        if prev:
            self.versions[prev-1]["stage"] = "Archived"
        self.versions[version-1]["stage"] = stage
        self.stages[stage] = version
        self.audit.append(f"v{version} -> {stage}" + (f" (v{prev} archived)" if prev else ""))

    def show(self):
        print(f"  {'ver':<5}{'registered':<13}{'test_acc':>9}{'latency_ms':>12}  {'stage':<12}{'data':<10}{'commit'}")
        print("  " + "-" * 78)
        for e in self.versions:
            print(f"  v{e['version']:<4}{e['registered']:<13}{e['metrics']['test_acc']:>9}"
                  f"{e['metrics']['latency_ms']:>12}  {e['stage']:<12}{e['data_hash']:<10}{e['code_commit']}")

reg = ModelRegistry()

print("--- Registering successive model versions ---\n")
reg.register("churn_model", {"test_acc": 0.842, "latency_ms": 21}, "d1a3f9", "a91c4e2")
reg.register("churn_model", {"test_acc": 0.871, "latency_ms": 24}, "d1a3f9", "b32d81f")
reg.register("churn_model", {"test_acc": 0.889, "latency_ms": 26}, "e77b21", "c4419aa")
reg.show()

print("\n--- Promote v2 to Production ---")
reg.transition(2, "Production")
reg.show()

print("\n--- v3 looks better offline: promote to Staging, then Production ---")
reg.transition(3, "Staging")
reg.transition(3, "Production")
reg.show()

print("\n--- INCIDENT: v3 degrades in production (data mismatch) ---")
print("  Live monitoring: accuracy dropped to 0.71, error rate up 3x")
print("  Action: roll back to the last known-good version (v2)\n")
reg.transition(2, "Production")
reg.show()

print("\n--- Audit trail (what regulators / postmortems need) ---")
for i, a in enumerate(reg.audit, 1):
    print(f"  {i}. {a}")

print("\n=== Why versioning matters ===")
print("  * Compare: a model trained in January differs from one trained in March")
print("  * Roll back quickly to a stable model, minimising downtime")
print("  * Reproducibility: model + data + code are versioned together")
print("  * Tools: Git (code), DVC / LakeFS (data), MLflow (models/metrics),")
print("           SageMaker Model Registry (cloud model versions)")

# Reproducibility check
print("\n=== Reproducibility check ===\n")
print("Re-running v2 with the SAME data hash and code commit should give")
print("the same metrics. If it does not, something untracked has changed.")
v2 = reg.versions[1]
rerun = {"test_acc": 0.871, "latency_ms": 24}
same = rerun == {k: v2["metrics"][k] for k in rerun}
print(f"  original v2 metrics : {v2['metrics']}")
print(f"  re-run metrics      : {rerun}")
print(f"  reproducible?       : {same}")
