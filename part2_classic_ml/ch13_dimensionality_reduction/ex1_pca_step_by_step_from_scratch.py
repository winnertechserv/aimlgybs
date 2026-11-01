"""
Chapter 13, Example 1: PCA Step-by-Step from Scratch
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This example builds PCA by hand, step by step, exactly as described in the overview. We
have 20 days of five weather readings, and one hidden weather pattern drives all of
them, so they are strongly correlated. Because the readings are in very different units
(Rain runs into hundreds of mm), we first standardize them, as Chapter 6 taught. Then we
compute the covariance matrix, find its eigenvalues and eigenvectors, sort them and
apply the 95% rule. PC1 alone captures about 97% of the variance, and its weights show
all five readings contributing almost equally: the five features really do carry one
shared signal. Finally we project the data onto PC1 (aₙ = Qᵀxₙ) and check that the
variance of the projected scores equals the PC1 eigenvalue.

Run it:           python ex1_pca_step_by_step_from_scratch.py
Expected output:  expected_output/ex1_pca_step_by_step_from_scratch.txt
"""
import numpy as np

np.random.seed(0)
N = 20
# One hidden 'weather' factor drives all five readings, plus a little noise.
# So the five features are strongly correlated - that is what PCA can exploit.
z = np.random.normal(0, 1, N)
data = np.column_stack([
    23.5 + 1.5 * z + np.random.normal(0, 0.25, N),   # Temperature (C)
    90.0 - 4.0 * z + np.random.normal(0, 0.70, N),   # Humidity (%)
    1012 - 6.0 * z + np.random.normal(0, 1.00, N),   # Pressure (hPa)
    600 - 400 * z + np.random.normal(0, 60.0, N),    # Rain (mm)
    50.0 - 15 * z + np.random.normal(0, 2.50, N),    # Moisture (%)
])
names = ["Temperature", "Humidity", "Pressure", "Rain", "Moisture"]
print(f"Dataset shape: {data.shape}  ({N} samples, 5 features)\n")

# Step 0: Standardize (Chapter 6). Rain is in hundreds of mm and Temperature in
# tens of degrees; without this step Rain's big units alone would decide PC1.
X_s = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
print("Step 0: Features standardized (mean 0, std 1)")

# Step 1: Covariance matrix of the standardized data (= correlation matrix)
cov = np.cov(X_s.T)                      # np.cov divides by N - 1
print("Step 1: Covariance matrix computed (5x5)")
print("        Correlation of Temperature with the others:", np.round(cov[0, 1:], 2))

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
    print(f"  PC{i+1}: {ev:>7.4f}  |  {ev/total*100:5.2f}%  |  cum {cum/total*100:6.2f}%{flag}")

# Step 3: Choose l with 95% criterion
l = next(i+1 for i in range(len(evals)) if evals[:i+1].sum()/total >= 0.95)
print(f"\nStep 3: Keep l = {l} principal component(s) (95% rule)")
pc1 = evecs[:, 0] * np.sign(evecs[0, 0])   # fix the sign so Temperature is positive
print("        PC1 weights:", {n: round(float(v), 2) for n, v in zip(names, pc1)})

# Step 4: Project data: a_n = Q^T x_n
Q = evecs[:, :l]
projected = X_s @ Q
print(f"Step 4: Original shape {data.shape} -> Projected shape {projected.shape}")
print(f"\nFirst 5 projected samples:")
for i in range(5):
    print(f"  Sample {i+1}: {projected[i, 0]:+.2f}")

# Step 5: the variance along PC1 equals its eigenvalue (both divide by N - 1)
print(f"\nStep 5: Variance of PC1 scores: {projected.var(ddof=1):.4f}"
      f"  (eigenvalue of PC1: {evals[0]:.4f})")
