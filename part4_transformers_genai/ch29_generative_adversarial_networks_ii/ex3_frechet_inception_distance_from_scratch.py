"""
Chapter 29, Example 3: Frechet Inception Distance from Scratch
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This builds FID directly from its definition, comparing the mean and covariance of real
and generated feature distributions. The experiments show FID near zero for a perfect
match, rising sharply for a shifted mean, and - importantly - also rising when variety
is too low, which is how FID detects the mode collapse that the Inception Score can
miss.

Run it:           python ex3_frechet_inception_distance_from_scratch.py
Expected output:  expected_output/ex3_frechet_inception_distance_from_scratch.txt
"""
# Chapter 29 - Example 3: Frechet Inception Distance (FID) from scratch
import numpy as np
from scipy.linalg import sqrtm
np.random.seed(2)

print("=== Frechet Inception Distance (FID) ===\n")
print("FID compares the DISTRIBUTIONS of Inception features for real and")
print("generated images, modelling each as a multivariate Gaussian:\n")
print("  FID = ||mu_r - mu_g||^2 + Tr( S_r + S_g - 2*(S_r S_g)^(1/2) )\n")
print("Lower FID = generated distribution closer to the real one.")
print("Unlike Inception Score, FID uses the REAL images as reference.\n")

def fid(f_real, f_gen):
    mu_r, mu_g = f_real.mean(axis=0), f_gen.mean(axis=0)
    S_r  = np.cov(f_real, rowvar=False)
    S_g  = np.cov(f_gen,  rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(S_r @ S_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(S_r + S_g - 2 * covmean))

n, dim = 500, 16
# Real image features
real = np.random.randn(n, dim)

experiments = {
  "Generated = real distribution (ideal)"   : np.random.randn(n, dim),
  "Slight mean shift (+0.3)"                : np.random.randn(n, dim) + 0.3,
  "Large mean shift (+1.5)"                 : np.random.randn(n, dim) + 1.5,
  "Correct mean, too little variety (0.4x)" : np.random.randn(n, dim) * 0.4,
  "Correct mean, too much variety (2.0x)"   : np.random.randn(n, dim) * 2.0,
}

print(f"Feature vectors: {n} images x {dim} dims (stand-in for Inception features)\n")
print(f"{'Generated-image distribution':<42} {'FID':>9}  {'Quality'}")
print("-" * 72)
for name, gen in experiments.items():
    score = fid(real, gen)
    q = ("excellent" if score < 1 else "good" if score < 5
         else "poor" if score < 20 else "very poor")
    print(f"  {name:<40} {score:>9.3f}  {q}")

print("\nObservations:")
print("  * Matching the real distribution gives FID near 0.")
print("  * A shifted mean (wrong content) raises FID sharply.")
print("  * Too little variety (mode collapse) ALSO raises FID, because the")
print("    covariance term no longer matches - FID penalises lost diversity.")

print("\n=== Inception Score vs FID ===\n")
print(f"  {'Aspect':<24} {'Inception Score':<26} {'FID'}")
print("  " + "-" * 70)
rows = [("Uses real images?",  "No",                    "Yes (reference set)"),
        ("Direction",          "Higher is better",      "Lower is better"),
        ("Detects mode collapse","Partially (marginal)", "Yes (covariance term)"),
        ("Based on",           "KL divergence",         "Frechet/Wasserstein-2 distance")]
for a, b, c in rows:
    print(f"  {a:<24} {b:<26} {c}")
