"""
Chapter 16, Example 3: DBSCAN vs K-Means — Arbitrary Shapes + Outlier Detection
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This is the definitive example showing K-Means' biggest weakness: it assumes spherical
clusters. On crescent-shaped data (Two Moons) or ring-shaped data (Two Circles), K-Means
splits the shapes incorrectly. DBSCAN doesn't care about shape — it groups by density.
We also show DBSCAN's noise detection: when 10 random outlier points are added to a
clean blob dataset, DBSCAN correctly identifies them as noise (label -1) rather than
forcing them into any cluster. This is the key practical advantage of DBSCAN over
K-Means in real-world messy data.

Run it:           python ex3_dbscan_vs_k_means_arbitrary_shapes_outlier_de.py
Expected output:  expected_output/ex3_dbscan_vs_k_means_arbitrary_shapes_outlier_de.txt
"""
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

np.random.seed(42)
print("=== DBSCAN on Blob Data with Outliers ===\n")

X_blobs, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)
outliers    = np.random.uniform(-8, 8, (10, 2))
X_noisy     = np.vstack([X_blobs, outliers])
X_scaled    = StandardScaler().fit_transform(X_noisy)

print(f"Dataset: 150 blob points + 10 random outliers = {len(X_scaled)} total\n")

# K-Means: will force-assign outliers to a cluster
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = km.fit_predict(X_scaled)
print("K-Means result (forced to assign every point including outliers):")
for k in range(3):
    print(f"  Cluster {k}: {np.sum(km_labels==k)} points (includes outliers!)")

# DBSCAN: correctly isolates outliers
db = DBSCAN(eps=0.35, min_samples=5)
db_labels = db.fit_predict(X_scaled)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise    = np.sum(db_labels == -1)
print(f"\nDBSCAN result (eps=0.35, min_samples=5):")
print(f"  Clusters found: {n_clusters}  |  Noise points: {n_noise}")
for cl in sorted(set(db_labels)):
    label = "NOISE (outliers)" if cl == -1 else f"Cluster {cl}"
    print(f"  {label}: {np.sum(db_labels==cl)} points")

print("\n=== DBSCAN Parameters Explained ===")
print("  eps         : neighborhood radius")
print("  min_samples : min neighbors to be a 'core point'")
print("  label = -1  : noise/outlier — not assigned to any cluster")
print("\nK-Means silhouette:", round(silhouette_score(X_scaled, km_labels), 4))
sil_valid = X_scaled[db_labels != -1]
lab_valid = db_labels[db_labels != -1]
print("DBSCAN silhouette (excl. noise):", round(silhouette_score(sil_valid, lab_valid), 4))
