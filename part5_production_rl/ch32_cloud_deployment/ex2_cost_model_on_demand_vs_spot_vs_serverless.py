"""
Chapter 32, Example 2: Cost Model - On-Demand vs Spot vs Serverless
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This quantifies the cost decisions described above. For a 200 GPU-hour training job,
spot instances save about 71% even after accounting for interruptions and checkpointing
overhead. For inference, it computes the break-even point between always-on containers
and serverless - below roughly 30 million requests a month serverless wins, above it a
container does.

Run it:           python ex2_cost_model_on_demand_vs_spot_vs_serverless.py
Expected output:  expected_output/ex2_cost_model_on_demand_vs_spot_vs_serverless.txt
"""
# Chapter 32 - Example 2: Cloud cost model - on-demand vs spot vs serverless
import numpy as np

print("=== Choosing Compute: On-Demand vs Spot vs Serverless ===\n")

# ---------- Training workload: GPU hours ----------
print("--- Workload A: TRAINING a deep model (fault-tolerant, batch) ---\n")
GPU_HOURS = 200
ON_DEMAND = 3.06          # $/GPU-hour
SPOT_DISCOUNT = 0.75      # spot is ~70-90% cheaper
SPOT = ON_DEMAND * (1 - SPOT_DISCOUNT)
INTERRUPT_RATE = 0.12     # fraction of hours lost to preemption
CHECKPOINT_OVERHEAD = 0.05

on_demand_cost = GPU_HOURS * ON_DEMAND
spot_hours = GPU_HOURS * (1 + INTERRUPT_RATE + CHECKPOINT_OVERHEAD)
spot_cost  = spot_hours * SPOT

print(f"  Job size: {GPU_HOURS} GPU-hours")
print(f"  {'Option':<34}{'rate $/h':>10}{'hours':>9}{'total':>11}")
print("  " + "-" * 64)
print(f"  {'On-demand GPU':<34}{ON_DEMAND:>10.2f}{GPU_HOURS:>9.0f}{on_demand_cost:>11.2f}")
print(f"  {'Spot / preemptible GPU':<34}{SPOT:>10.2f}{spot_hours:>9.0f}{spot_cost:>11.2f}")
print(f"\n  Saving with spot: ${on_demand_cost - spot_cost:,.2f} "
      f"({(1 - spot_cost/on_demand_cost)*100:.0f}%)")
print("  Spot requires checkpointing: interrupted work must resume, not restart.")
print("  Suitable for batch training, ETL and hyperparameter tuning;")
print("  NOT suitable for latency-critical production inference.\n")

# ---------- Inference workload: containers vs serverless ----------
print("--- Workload B: INFERENCE endpoint (latency sensitive) ---\n")
CONTAINER_HOUR = 0.096            # small always-on container instance
LAMBDA_PER_REQ = 0.0000002        # per invocation
LAMBDA_GB_SEC  = 0.0000166667     # per GB-second
MEM_GB, DURATION_S = 0.5, 0.25    # per inference call

print(f"  {'Requests/month':>16}{'Always-on container':>22}{'Serverless (FaaS)':>20}  {'Cheaper'}")
print("  " + "-" * 78)
for reqs in [10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]:
    container = CONTAINER_HOUR * 24 * 30                       # always running
    serverless = reqs * LAMBDA_PER_REQ + reqs * MEM_GB * DURATION_S * LAMBDA_GB_SEC
    cheaper = "serverless" if serverless < container else "container"
    print(f"  {reqs:>16,}{'$'+format(container,'.2f'):>22}{'$'+format(serverless,'.2f'):>20}  {cheaper}")

break_even = (CONTAINER_HOUR*24*30) / (LAMBDA_PER_REQ + MEM_GB*DURATION_S*LAMBDA_GB_SEC)
print(f"\n  Break-even point: about {break_even:,.0f} requests/month")
print("  Below it, serverless is cheaper (scale-to-zero when idle).")
print("  Above it, an always-on container is cheaper and avoids cold starts.\n")

# ---------- Decision guidance ----------
print("=== Choosing the right service ===\n")
rows = [
 ("Large distributed training", "GPU/TPU nodes on SageMaker, Vertex AI, Azure ML"),
 ("Custom model serving",       "Containers on GKE / EKS / AKS or ECS-Fargate"),
 ("Lightweight inference",      "Serverless: AWS Lambda, Cloud Functions, Azure Functions"),
 ("Feature storage & serving",  "BigQuery, DynamoDB, Cosmos DB"),
 ("Cost-sensitive batch jobs",  "Spot / preemptible instances with checkpointing"),
]
print(f"  {'Need':<30}{'Recommended'}")
print("  " + "-" * 74)
for need, rec in rows:
    print(f"  {need:<30}{rec}")

print("\n  Serverless limits: execution time caps, limited memory/CPU, no GPU,")
print("  cold-start latency, and statelessness - so it cannot train large models.")
