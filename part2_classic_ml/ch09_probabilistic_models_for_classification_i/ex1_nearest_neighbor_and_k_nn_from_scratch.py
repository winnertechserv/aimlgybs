"""
Chapter 9, Example 1: Nearest Neighbor and K-NN from Scratch
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This follows the four steps: compute distances, sort, take the first K, then vote. For
the test person (150 cm, 47 kg) the four nearest examples are all adults, at distances
5.39, 8.73, 18.48 and 19.58. For a second person (140 cm, 46 kg), K = 1 and K = 3 say
'Adult' but K = 5 says 'Child'. The choice of K really can change the answer. The last
part picks K for the Iris flowers using a validation set.

Run it:           python ex1_nearest_neighbor_and_k_nn_from_scratch.py
Expected output:  expected_output/ex1_nearest_neighbor_and_k_nn_from_scratch.txt
"""
# Chapter 9 - Example 1: Nearest neighbor and K-NN from scratch (adult vs child)
import numpy as np
from collections import Counter

X = np.array([[98, 16.4], [104, 15.9], [110, 19.8], [115, 19.1], [119, 23.0],
              [123, 22.4], [127, 27.1], [131, 26.3], [134, 31.0], [101, 17.5],
              [152, 52.0], [158, 50.5], [161, 63.2], [165, 57.8], [168, 71.0],
              [172, 64.5], [175, 80.3], [179, 72.9], [183, 88.0], [188, 81.4]])
y = np.r_[np.zeros(10, int), np.ones(10, int)]
NAMES = {0: "Child", 1: "Adult"}

# Raw cm and kg are used so the numbers match Figure 9.1. In general, standardize the
# features before K-NN (Chapter 6), as Example 2 does.
def knn_predict(x, K):
    d = np.sqrt(((X - x) ** 2).sum(axis=1))        # Step 1: Euclidean distance to all
    order = np.argsort(d)                           # Step 2: sort, nearest first
    nearest = order[:K]                             # Step 3: take the first K
    vote = Counter(y[nearest]).most_common(1)[0][0] # Step 4: majority vote
    return vote, d, order

for test in ([150, 47.0], [140, 46.0]):
    print(f"Test person: height {test[0]} cm, weight {test[1]} kg")
    _, d, order = knn_predict(np.array(test), 1)
    print("  5 nearest training examples:")
    for i in order[:5]:
        print(f"    ({X[i, 0]:.0f}, {X[i, 1]:5.2f})  distance {d[i]:6.2f}  class {NAMES[y[i]]}")
    for K in (1, 3, 5):
        print(f"  K = {K}: predicted {NAMES[knn_predict(np.array(test), K)[0]]}")
    print()

# Choosing K with a validation set (Iris flowers: 150 samples, 3 classes)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
Xi, yi = load_iris(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(Xi, yi, test_size=0.4, stratify=yi, random_state=3)
print("Choosing K on a validation set (Iris):")
for K in (1, 3, 5, 7, 9, 15, 25):
    acc = KNeighborsClassifier(n_neighbors=K).fit(X_tr, y_tr).score(X_val, y_val)
    print(f"  K = {K:2d}: validation accuracy {acc:.3f}")
