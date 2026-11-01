"""
Chapter 13, Example 2: PCA with scikit-learn on Handwritten Digits
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This is a real-world PCA example on the famous handwritten digits dataset — the same
idea scales to larger 28×28 images. Each image is 8×8 = 64 pixels. We use sklearn's PCA
to find how many principal components are needed to capture 95% and 99% of the variance.
The output shows that just 40 out of 64 dimensions preserve 95% of information — a 37.5%
size reduction. The reconstruction error (MSE) is tiny, confirming that the 24 discarded
dimensions were mostly noise. This is exactly what the eigenface compression concept
shows — you store a few component scores per image instead of all raw pixel values.

Run it:           python ex2_pca_with_scikit_learn_on_handwritten_digits.py
Expected output:  expected_output/ex2_pca_with_scikit_learn_on_handwritten_digits.txt
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
X = digits.data   # shape: (1797, 64)

print(f"Original dataset shape : {X.shape}")
print(f"  -> 1797 images, each 8x8 = 64 pixel features\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA()
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

l_95 = np.argmax(cumvar >= 95) + 1
l_99 = np.argmax(cumvar >= 99) + 1

print("Variance explained by top N components:")
for i in [0, 1, 2, 4, 9, 19, l_95-1, l_99-1]:
    print(f"  Top {i+1:>3} PCs -> cumulative variance: {cumvar[i]:.2f}%")

print(f"\n95% threshold: keep {l_95} components")
print(f"99% threshold: keep {l_99} components")
print(f"Reduction: 64 -> {l_95} dims  ({(1 - l_95/64)*100:.1f}% smaller)")

pca = PCA(n_components=l_95)
X_reduced = pca.fit_transform(X_scaled)
X_recon   = pca.inverse_transform(X_reduced)
mse = np.mean((X_scaled - X_recon) ** 2)

print(f"\nReduced dataset shape  : {X_reduced.shape}")
print(f"Reconstruction MSE     : {mse:.6f}")
print("(Small MSE = minimal information lost)")
