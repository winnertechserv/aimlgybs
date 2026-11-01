"""
Chapter 20, Example 2: Batch Normalization — Manual Calculation
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We compute Batch Normalization step by step on a small hidden layer: 4 neurons, mini-
batch of 5 examples. Step 1 computes the mean and variance per neuron across all 5
examples (column-wise). Step 2 subtracts the mean and divides by standard deviation —
producing s_hat with mean=0, std=1 for every neuron. This is verified explicitly. Step 3
applies the learnable γ and β parameters (scale and shift), giving the final output z.
The last section proves why BatchNorm is needed: two mini-batches with completely
different raw distributions (means 2.04 vs 7.88) both become mean=0, std=1 after
normalization — the internal covariate shift is eliminated.

Run it:           python ex2_batch_normalization_manual_calculation.py
Expected output:  expected_output/ex2_batch_normalization_manual_calculation.txt
"""
import numpy as np

np.random.seed(42)
s = np.array([[2.1,-0.5,3.2,0.8],
              [1.8, 0.3,3.5,1.2],
              [2.5,-0.9,2.8,0.5],
              [1.5, 0.7,3.0,0.9],
              [2.3,-0.2,3.4,1.1]])  # 5 examples x 4 neurons

print("Hidden layer output s (5 examples x 4 neurons):")
print(s)

# Step 1: mean and variance across mini-batch (per neuron = column-wise)
mu    = s.mean(axis=0)
var   = s.var(axis=0)
sigma = np.sqrt(var + 1e-8)
print(f"\nStep 1: Per-neuron statistics (computed across all 5 examples):")
print(f"  mean mu   = {mu.round(4)}")
print(f"  var  s^2  = {var.round(4)}")
print(f"  std  sigma= {sigma.round(4)}")

# Step 2: normalize
s_hat = (s - mu) / sigma
print(f"\nStep 2: Normalized s_hat = (s - mu) / sigma:")
print(s_hat.round(4))
print(f"  Verify mean : {s_hat.mean(axis=0).round(6)}  <- all ~0")
print(f"  Verify std  : {s_hat.std(axis=0).round(6)}  <- all ~1")

# Step 3: scale and shift with learnable parameters
gamma = np.array([1.5, 0.8, 2.0, 1.2])
beta  = np.array([0.1, 0.0, -0.5, 0.3])
z = gamma * s_hat + beta
print(f"\nStep 3: z = gamma * s_hat + beta")
print(f"  gamma={gamma}  beta={beta}")
print(f"Final output z:\n{z.round(4)}")

print("\n=== Internal Covariate Shift Demo ===\n")
mb1 = np.array([2.1,1.8,2.5,1.5,2.3])  # mini-batch 1 outputs (neuron 1)
mb2 = np.array([8.4,7.2,9.0,6.0,8.8])  # mini-batch 2 outputs (same neuron, shifted)
bn1 = (mb1 - mb1.mean()) / mb1.std()
bn2 = (mb2 - mb2.mean()) / mb2.std()
print(f"Mini-batch 1 raw: mean={mb1.mean():.2f}, std={mb1.std():.2f}")
print(f"Mini-batch 2 raw: mean={mb2.mean():.2f}, std={mb2.std():.2f}")
print(f"After BN batch 1: mean={bn1.mean():.2f}, std={bn1.std():.2f}")
print(f"After BN batch 2: mean={bn2.mean():.2f}, std={bn2.std():.2f}")
print("\nBoth batches normalized to same distribution -> covariate shift eliminated!")
