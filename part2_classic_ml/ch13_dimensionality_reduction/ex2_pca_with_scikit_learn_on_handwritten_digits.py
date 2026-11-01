"""
Chapter 13, Example 2: PCA with scikit-learn on Handwritten Digits
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This is PCA on real data: scikit-learn's handwritten digits, where each image is 8×8 =
64 pixels (the same idea scales to 28×28 MNIST images). We use scikit-learn's PCA to
find how many components are needed to keep 95% and 99% of the variance. All pixels
share one unit (brightness from 0 to 16), so we do not standardize them. Just 29 of the
64 dimensions keep 95% of the variance, matching Figure 13.2, and the images shrink by
more than half. The reconstruction error is small next to the average pixel variance, so
the 35 dropped directions carried little information. This is the eigenface idea in
action: store a few component scores per image instead of every pixel.

Run it:           python ex2_pca_with_scikit_learn_on_handwritten_digits.py
Expected output:  expected_output/ex2_pca_with_scikit_learn_on_handwritten_digits.txt
"""
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
X = digits.data   # shape: (1797, 64), pixel brightness 0-16

print(f"Original dataset shape : {X.shape}")
print(f"  -> 1797 images, each 8x8 = 64 pixel features\n")

# All pixels share one unit (brightness 0-16), so we do NOT standardize here.
# Standardizing would blow up the near-constant border pixels to the same weight
# as the useful ones. (Note: we fit on all images because this is unsupervised
# compression; before a classifier, fit PCA on the training images only, as in Example 3.)
pca_full = PCA()
pca_full.fit(X)
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
X_reduced = pca.fit_transform(X)
X_recon   = pca.inverse_transform(X_reduced)
mse = np.mean((X - X_recon) ** 2)

print(f"\nReduced dataset shape  : {X_reduced.shape}")
print(f"Reconstruction MSE     : {mse:.4f}  (average pixel variance: {X.var(axis=0).mean():.2f})")
print("(Small MSE = minimal information lost)")
