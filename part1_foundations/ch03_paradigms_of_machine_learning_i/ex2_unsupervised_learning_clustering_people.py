"""
Chapter 3, Example 2: Unsupervised Learning — Clustering People
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Here K-means gets the same 20 people but no labels at all. The table shows compactness
(inertia) for 1 to 5 clusters. The big drop from 1 to 2 clusters is the 'elbow', so two
groups is a natural choice. The two cluster centers come out at exactly the average
child (116.2 cm, 21.85 kg) and the average adult (170.1 cm, 68.16 kg). The model found
children and adults without ever being told they exist.

Run it:           python ex2_unsupervised_learning_clustering_people.py
Expected output:  expected_output/ex2_unsupervised_learning_clustering_people.txt
"""
# Chapter 3 - Example 2: Unsupervised learning - clustering WITHOUT labels
# K-means groups people by height and weight. We never show it who is a child or an adult.
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[98, 16.4], [104, 15.9], [110, 19.8], [115, 19.1], [119, 23.0],
              [123, 22.4], [127, 27.1], [131, 26.3], [134, 31.0], [101, 17.5],
              [152, 52.0], [158, 50.5], [161, 63.2], [165, 57.8], [168, 71.0],
              [172, 64.5], [175, 80.3], [179, 72.9], [183, 88.0], [188, 81.4]])
hidden_truth = np.r_[np.zeros(10, int), np.ones(10, int)]   # only used to CHECK at the end

print(" k   compactness (inertia)")
for k in range(1, 6):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(f" {k}   {km.inertia_:10.1f}")
print("Compactness drops sharply from k=1 to k=2, then flattens (the 'elbow').")

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
print("\nCluster centers (height, weight):")
for c in km.cluster_centers_:
    print(f"  ({c[0]:.1f} cm, {c[1]:.2f} kg)")

# Cluster numbers are arbitrary, so match them to the hidden labels before comparing
labels = km.labels_
if np.mean(labels == hidden_truth) < 0.5:
    labels = 1 - labels
print(f"\nAgreement with the hidden child/adult labels: {np.mean(labels == hidden_truth):.0%}")
print("New person (130 cm, 30 kg) joins cluster with center",
      np.round(km.cluster_centers_[km.predict([[130, 30]])[0]], 1))
