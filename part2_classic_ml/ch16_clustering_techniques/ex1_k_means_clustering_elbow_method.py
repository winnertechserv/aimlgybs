"""
Chapter 16, Example 1: K-Means Clustering + Elbow Method
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We run K-Means on a synthetic 4-cluster dataset and use two methods to find the optimal
K. The Elbow Method plots distortion (J) vs K — the biggest drop is between K=1 and K=4,
after which improvements become marginal. The Silhouette Score (which measures how well-
separated clusters are, ranging from -1 to 1) peaks at K=4 with 0.8335, independently
confirming the elbow. The final K=4 cluster centers and per-cluster counts are also
printed, showing balanced clusters of 75 points each.

Run it:           python ex1_k_means_clustering_elbow_method.py
Expected output:  expected_output/ex1_k_means_clustering_elbow_method.txt
"""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import numpy as np

np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
print(f"Dataset: {X.shape[0]} samples | Ground truth: 4 clusters\n")

print("=== Elbow Method ===\n")
print(f"{'K':>4}  {'Distortion J':>14}  {'Change':>10}  {'Silhouette':>11}")
print("-" * 46)
prev = None
for k in range(1, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    J      = km.inertia_
    change = f"{J-prev:.1f}" if prev else "—"
    sil    = f"{silhouette_score(X, km.labels_):.4f}" if k >= 2 else "N/A"
    prev   = J
    print(f"{k:>4}  {J:>14.2f}  {change:>10}  {sil:>11}")

print()
print("Elbow at K=4 — biggest drop plateaus after K=4.")
print("Silhouette also peaks at K=4. Both methods agree!\n")

km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
km4.fit(X)
print("=== K-Means Result (K=4) ===\n")
print("Cluster Centers:")
for i, c in enumerate(km4.cluster_centers_):
    n = np.sum(km4.labels_==i)
    print(f"  Cluster {i}: ({c[0]:+.3f}, {c[1]:+.3f}) | {n} points")
print(f"\nTotal Distortion J = {km4.inertia_:.4f}")
print(f"Silhouette Score   = {silhouette_score(X, km4.labels_):.4f}")
