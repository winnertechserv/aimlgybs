"""
Chapter 13, Example 1: PCA Step-by-Step from Scratch
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This example builds PCA manually without sklearn — step by step, exactly as described in
the overview. We have a 20-sample, 5-feature atmospheric dataset. First we compute the
covariance matrix, then perform eigendecomposition to find the principal directions. We
sort eigenvalues in descending order (most important direction first) and apply the 95%
variance criterion to choose how many dimensions to keep. Notice that PC1 alone captures
99.73% of variance — the data is essentially 1-dimensional despite having 5 features!
Finally we project the 5D data into 1D and check the output is uncorrelated.

Run it:           python ex1_pca_step_by_step_from_scratch.py
Expected output:  expected_output/ex1_pca_step_by_step_from_scratch.txt
"""
import numpy as np

np.random.seed(0)
N = 20
data = np.column_stack([
    np.random.uniform(20, 27, N),    # Temperature
    np.random.uniform(82, 99, N),    # Humidity
    np.random.uniform(990, 1040, N), # Pressure
    np.random.uniform(0, 1800, N),   # Rain
    np.random.uniform(0, 100, N),    # Moisture
])

print(f"Dataset shape: {data.shape}  ({N} samples, 5 features)\n")

# Step 1: Covariance matrix
mean = data.mean(axis=0)
X_c  = data - mean
cov  = np.cov(X_c.T)
print("Step 1: Covariance matrix computed (5x5)")

# Step 2: Eigen decomposition (sorted descending)
evals, evecs = np.linalg.eigh(cov)
idx    = np.argsort(evals)[::-1]
evals  = evals[idx]
evecs  = evecs[:, idx]

print("\nStep 2: Eigenvalues and cumulative variance:")
total = evals.sum()
cum   = 0
for i, ev in enumerate(evals):
    cum += ev
    flag = " <- 95% crossed" if cum/total >= 0.95 and (cum-ev)/total < 0.95 else ""
    print(f"  PC{i+1}: {ev:>12.2f}  |  {ev/total*100:5.2f}%  |  cum {cum/total*100:5.2f}%{flag}")

# Step 3: Choose l with 95% criterion
l = next(i+1 for i in range(len(evals)) if evals[:i+1].sum()/total >= 0.95)
print(f"\nStep 3: Keep l = {l} principal component(s) (95% rule)")

# Step 4: Project data
Q = evecs[:, :l]
projected = X_c @ Q
print(f"Step 4: Original shape {data.shape} -> Projected shape {projected.shape}")
print(f"\nFirst 5 projected samples:")
for i in range(5):
    print(f"  Sample {i+1}: {np.round(projected[i], 2)}")

print(f"\nStep 5: Variance of PC1: {projected.var():.4f}")
print("        (equals eigenvalue of PC1 - data is uncorrelated in new space)")
