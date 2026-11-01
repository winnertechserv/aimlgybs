"""
Chapter 19, Example 3: AdaGrad vs RMSProp — Learning Rate Decay Comparison
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This example simulates 30 steps of gradient descent and traces the effective learning
rate of AdaGrad vs RMSProp for a single weight. The first 10 steps have large gradients
(simulating early training); the last 20 have small gradients (simulating late training
near convergence). AdaGrad accumulates ALL squared gradients — so by step 11, its
effective learning rate has already decayed to 0.00389 and keeps shrinking even as
gradients become small. RMSProp only keeps a moving average (ρ=0.9), so when gradients
become small, its denominator shrinks too, keeping the learning rate healthy. The Ratio
column shows RMSProp learning 11x faster than AdaGrad by step 30.

Run it:           python ex3_adagrad_vs_rmsprop_learning_rate_decay_compar.py
Expected output:  expected_output/ex3_adagrad_vs_rmsprop_learning_rate_decay_compar.txt
"""
import numpy as np

np.random.seed(0)
T   = 30
eta = 0.1
rho = 0.9
eps = 1e-8

# Simulate gradients: large in first 10 steps, small afterward
grads = np.concatenate([np.random.uniform(5, 10, 10),
                         np.random.uniform(0.1, 0.5, 20)])

G_ada = 0.0
v_rms = 0.0

print("=== AdaGrad vs RMSProp: Effective Learning Rate Trace ===\n")
print("Steps 1-10: large gradients | Steps 11-30: small gradients\n")
print(f"{'Step':>5} {'Gradient':>10} {'AdaGrad LR':>12} {'RMSProp LR':>12}  {'Ratio'}")
print("-" * 55)
for t in range(1, T + 1):
    g = grads[t - 1]
    G_ada += g**2
    lr_ada = eta / (np.sqrt(G_ada) + eps)
    v_rms  = rho * v_rms + (1 - rho) * g**2
    lr_rms = eta / (np.sqrt(v_rms) + eps)
    ratio  = lr_rms / lr_ada
    print(f"{t:>5} {g:>10.4f} {lr_ada:>12.6f} {lr_rms:>12.6f}  {ratio:>5.1f}x")

print()
print("Key insight:")
print("  AdaGrad LR decays to ~0.0039 and stays there even for tiny gradients.")
print("  RMSProp LR recovers because it forgets old (large) gradients over time.")
print("  By step 30, RMSProp learns 11x faster than AdaGrad on the same data!")
