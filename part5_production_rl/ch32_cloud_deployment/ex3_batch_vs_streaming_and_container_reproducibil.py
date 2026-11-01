"""
Chapter 32, Example 3: Batch vs Streaming, and Container Reproducibility
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
The first half compares batch and streaming architectures on latency and throughput and
simulates a traffic burst, showing how a backlog forms when arrivals exceed capacity -
the exact problem autoscaling solves. The second half shows how a Dockerfile pins the
whole environment so that three machines that would otherwise drift on Python, numpy and
sklearn versions become identical.

Run it:           python ex3_batch_vs_streaming_and_container_reproducibil.py
Expected output:  expected_output/ex3_batch_vs_streaming_and_container_reproducibil.txt
"""
# Chapter 32 - Example 3: Batch vs Streaming architectures + container reproducibility
import numpy as np, hashlib
np.random.seed(1)

print("=== Batch vs Streaming Architectures ===\n")
print("Batch     : process large volumes at scheduled intervals (retraining, ETL)")
print("Streaming : process records in real time as they arrive (fraud, IoT)\n")

N_EVENTS = 10_000
EVENT_RATE = 500          # events per second arriving
BATCH_INTERVAL_S = 3600   # batch job runs hourly
BATCH_THROUGHPUT = 5000   # records/sec when the batch job runs
STREAM_THROUGHPUT = 600   # records/sec continuous
STREAM_PER_REC_MS = 12

# ---- Batch ----
batch_runtime = N_EVENTS / BATCH_THROUGHPUT
avg_wait = BATCH_INTERVAL_S / 2                 # average time an event waits
batch_latency = avg_wait + batch_runtime
# ---- Streaming ----
stream_latency = STREAM_PER_REC_MS / 1000.0

print(f"Workload: {N_EVENTS:,} events arriving at {EVENT_RATE}/s\n")
print(f"  {'Metric':<34}{'Batch':>16}{'Streaming':>16}")
print("  " + "-" * 66)
print(f"  {'Processing trigger':<34}{'every 60 min':>16}{'per event':>16}")
print(f"  {'End-to-end latency':<34}{batch_latency:>13.1f} s{stream_latency*1000:>13.0f} ms")
print(f"  {'Throughput':<34}{BATCH_THROUGHPUT:>13,}/s{STREAM_THROUGHPUT:>13,}/s")
print(f"  {'Compute utilisation':<34}{'bursty':>16}{'continuous':>16}")
print(f"  {'Relative cost':<34}{'lower':>16}{'higher':>16}")

print(f"\n  Latency difference: {batch_latency/stream_latency:,.0f}x")
print("\n  Choose BATCH     : scheduled retraining, historical scoring, ETL")
print("  Choose STREAMING : fraud detection, live recommendations, IoT analytics")
print("  Tools for streaming: Pub/Sub, Kinesis, Event Hub")
print("  HYBRID is common: stream for real-time decisions, batch for retraining.\n")

# ---- Simulate a streaming queue under a traffic burst ----
print("--- Streaming queue behaviour during a burst ---\n")
arrival = np.array([300, 350, 900, 1200, 1100, 400, 320, 300])   # events/s per window
queue = 0
print(f"  {'window':>7}{'arrivals/s':>12}{'capacity/s':>12}{'queue':>8}{'latency':>11}")
print("  " + "-" * 52)
for i, a in enumerate(arrival, 1):
    queue = max(0, queue + a - STREAM_THROUGHPUT)
    lat = STREAM_PER_REC_MS + (queue / STREAM_THROUGHPUT) * 1000
    flag = "  <-- backlog" if queue > 0 else ""
    print(f"  {i:>7}{a:>12}{STREAM_THROUGHPUT:>12}{queue:>8}{lat:>9.0f}ms{flag}")
print("\n  A sustained burst above capacity builds a backlog and latency grows.")
print("  This is exactly what autoscaling (adding consumers) is meant to fix.\n")

# ---- Container reproducibility ----
print("=== Containerisation: solving 'works on my machine' ===\n")
dockerfile = """FROM python:3.10-slim
RUN pip install numpy==2.4.4 scikit-learn==1.8.0 pandas==3.0.2
COPY model.pkl /app/model.pkl
COPY serve.py  /app/serve.py
CMD ["python", "/app/serve.py"]"""
print("Dockerfile (pins the ENTIRE environment):")
for line in dockerfile.split("\n"):
    print(f"    {line}")

img = hashlib.sha256(dockerfile.encode()).hexdigest()[:16]
print(f"\n  Image digest: sha256:{img}")
print("  Same Dockerfile -> same digest -> identical environment everywhere.\n")

envs = {
  "Developer laptop (no container)": {"python": "3.11.2", "numpy": "2.1.0", "sklearn": "1.6.0"},
  "CI server (no container)"       : {"python": "3.9.16", "numpy": "2.4.4", "sklearn": "1.8.0"},
  "Production VM (no container)"   : {"python": "3.10.9", "numpy": "2.0.1", "sklearn": "1.7.1"},
}
pinned = {"python": "3.10", "numpy": "2.4.4", "sklearn": "1.8.0"}
print(f"  {'Environment':<34}{'python':>9}{'numpy':>9}{'sklearn':>10}  {'Matches pinned?'}")
print("  " + "-" * 78)
for name, e in envs.items():
    ok = all(e[k].startswith(pinned[k]) for k in pinned)
    print(f"  {name:<34}{e['python']:>9}{e['numpy']:>9}{e['sklearn']:>10}  {'yes' if ok else 'NO - drift'}")
print(f"  {'Any machine WITH the container':<34}{pinned['python']:>9}{pinned['numpy']:>9}{pinned['sklearn']:>10}  yes")

print("\n  Containers bundle code + dependencies + runtime into one image, so")
print("  CUDA/cuDNN versions, library versions and the OS layer stay identical")
print("  across training, CI, and production.")
