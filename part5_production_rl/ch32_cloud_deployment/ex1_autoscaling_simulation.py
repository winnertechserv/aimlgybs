"""
Chapter 32, Example 1: Autoscaling Simulation
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This runs the Kubernetes HPA control loop over a realistic 24-hour traffic curve:
monitor utilization, compare with the target, and adjust replica count. The comparison
at the end quantifies the trade-off - autoscaling holds latency within SLA at 55% of the
cost of permanent over-provisioning, whereas fixed capacity is cheaper but breaches the
SLA in 14 of 24 hours.

Run it:           python ex1_autoscaling_simulation.py
Expected output:  expected_output/ex1_autoscaling_simulation.txt
"""
# Chapter 32 - Example 1: Autoscaling simulation (Kubernetes HPA style)
import numpy as np
np.random.seed(0)

print("=== Cloud Autoscaling for ML Inference ===\n")
print("Scalability = adjust capacity to match demand while keeping latency")
print("stable. The control loop:")
print("  monitor metrics -> evaluate thresholds -> adjust capacity -> stabilize\n")

TARGET_CPU   = 0.70     # target utilization per replica
CAP_PER_POD  = 100      # requests/sec one replica can serve
MIN_PODS, MAX_PODS = 2, 20
BASE_LAT     = 40       # ms at low load

# A day of traffic: quiet night, morning ramp, lunch peak, evening spike
hours = np.arange(24)
traffic = np.array([120,90,70,60,70,110,220,420,650,780,830,910,
                    1180,980,860,820,900,1150,1320,1080,760,520,320,180])

def latency(load_ratio):
    """Latency grows sharply as utilization approaches 1 (queueing)."""
    return BASE_LAT / max(1e-3, (1 - min(load_ratio, 0.99)))

print(f"Config: {CAP_PER_POD} req/s per replica, target CPU {TARGET_CPU:.0%}, "
      f"min={MIN_PODS}, max={MAX_PODS} replicas\n")
print(f"{'hour':>5}{'req/s':>8}{'replicas':>10}{'util':>8}{'latency':>10}  {'action'}")
print("-" * 62)

pods = MIN_PODS
fixed_pods = 6            # for the comparison below
total_pod_hours, sla_breaches = 0, 0
fixed_breaches = 0

for h, rps in zip(hours, traffic):
    # --- HPA formula: desired = ceil(current * (currentUtil / targetUtil)) ---
    # (Simplified: real HPA also skips scaling while the ratio is within 10% of 1,
    #  a tolerance band that stops it flapping when usage hovers near the target.)
    util_now = rps / (pods * CAP_PER_POD)
    desired  = int(np.ceil(pods * (util_now / TARGET_CPU)))
    desired  = max(MIN_PODS, min(MAX_PODS, desired))
    action   = ("scale UP" if desired > pods else
                "scale DOWN" if desired < pods else "hold")
    pods = desired

    util = rps / (pods * CAP_PER_POD)
    lat  = latency(util)
    total_pod_hours += pods
    if lat > 200: sla_breaches += 1

    fixed_util = rps / (fixed_pods * CAP_PER_POD)
    if latency(fixed_util) > 200: fixed_breaches += 1

    if h % 3 == 0 or h in (12, 18):
        print(f"{h:>5}{rps:>8}{pods:>10}{util:>8.0%}{lat:>9.0f}ms  {action}")

# ---- Cost comparison ----
POD_HOUR = 0.10
print("\n=== Autoscaling vs Fixed Capacity (24 hours) ===\n")
fixed_hours = fixed_pods * 24
over_hours  = MAX_PODS * 24
print(f"{'Strategy':<34}{'pod-hours':>11}{'cost':>10}{'SLA breaches':>14}")
print("-" * 70)
print(f"  {'Autoscaling (2-20 replicas)':<32}{total_pod_hours:>11}${total_pod_hours*POD_HOUR:>9.2f}{sla_breaches:>14}")
print(f"  {'Fixed 6 replicas':<32}{fixed_hours:>11}${fixed_hours*POD_HOUR:>9.2f}{fixed_breaches:>14}")
print(f"  {'Over-provisioned 20 replicas':<32}{over_hours:>11}${over_hours*POD_HOUR:>9.2f}{0:>14}")

saving = (over_hours - total_pod_hours) / over_hours * 100
print(f"\nAutoscaling saves {saving:.0f}% versus permanent over-provisioning,")
print(f"while keeping SLA breaches at {sla_breaches} (fixed capacity: {fixed_breaches}).")

print("\n=== Types of autoscaling ===")
print("  Instance autoscaling : VM level - EC2 Auto Scaling Groups, GCE Instance Groups")
print("  Container autoscaling: pod level - Kubernetes HPA (horizontal) and VPA (vertical)")
print("  Serverless autoscaling: function level - scales to ZERO when idle, pay per call")
print("\nHorizontal scaling adds machines/pods; vertical scaling grows one machine")
print("(more CPU/RAM/GPU) and is limited by the largest available hardware.")
