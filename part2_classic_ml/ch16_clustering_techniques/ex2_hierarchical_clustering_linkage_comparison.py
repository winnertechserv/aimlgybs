"""
Chapter 16, Example 2: Hierarchical Clustering & Linkage Comparison
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Hierarchical clustering does not need you to specify K in advance — it builds a full
tree of merges (a dendrogram) and you cut it at any level you want. We compare four
linkage methods on a 3-cluster dataset. All four linkages perfectly recover the true
clusters here (ARI=1.0). The merge history table shows the actual step-by-step merging
process: step 1 merges points #16 and #53 (distance 0.0484), step 2 merges #57 and #85,
and so on, until all 100 points are in one cluster after 99 merges. This is exactly the
AGNES algorithm described in the overview.

Run it:           python ex2_hierarchical_clustering_linkage_comparison.py
Expected output:  expected_output/ex2_hierarchical_clustering_linkage_comparison.txt
"""
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage
import numpy as np

np.random.seed(42)
X, y_true = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)
print(f"Dataset: {X.shape[0]} samples | Ground truth: 3 clusters\n")
print(f"{'Linkage':<20} {'Silhouette':>12} {'Adj Rand':>12}  Notes")
print("-" * 75)

for lnk in ['single', 'complete', 'average', 'ward']:
    agg    = AgglomerativeClustering(n_clusters=3, linkage=lnk)
    labels = agg.fit_predict(X)
    sil    = silhouette_score(X, labels)
    ari    = adjusted_rand_score(y_true, labels)
    note   = {'single':'nearest-neighbor chaining','complete':'compact clusters',
              'average':'balanced choice','ward':'minimizes within-cluster variance'}[lnk]
    print(f"  {lnk:<18} {sil:>12.4f} {ari:>12.4f}  {note}")

print("\n=== Merge History (AGNES, Ward linkage, first 8 steps) ===\n")
print(f"{'Step':>5} {'Cluster A':>10} {'Cluster B':>10} {'Distance':>10} {'New Size':>10}")
print("-" * 50)
Z = linkage(X, method='ward')
n = len(X)
for step, (ca, cb, dist, size) in enumerate(Z[:8], 1):
    a = f"#{int(ca)}" if ca < n else f"merged_{int(ca)-n+1}"
    b = f"#{int(cb)}" if cb < n else f"merged_{int(cb)-n+1}"
    print(f"{step:>5} {a:>10} {b:>10} {dist:>10.4f} {int(size):>10}")
print(f"  ...")
print(f"\nTotal steps: {n-1}  (from {n} singletons -> 1 cluster)")
