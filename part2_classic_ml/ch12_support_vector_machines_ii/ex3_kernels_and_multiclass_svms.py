"""
Chapter 12, Example 3: Kernels and Multiclass SVMs
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
First, the kernel trick in numbers: (x·z + 1)² gives 25, and building the 6-dimensional
Φ(x) and Φ(z) explicitly and taking their dot product also gives 25. Same answer, less
work. Next, two rings of points that no line can separate. A linear SVM manages only
54%, while polynomial and Gaussian kernels both reach 100%. Finally, a 4-class problem:
one-against-the-rest trains 4 SVMs, one-against-one trains 6 (= 4×3/2), and both reach
similar accuracy.

Run it:           python ex3_kernels_and_multiclass_svms.py
Expected output:  expected_output/ex3_kernels_and_multiclass_svms.txt
"""
# Chapter 12 - Example 3: Kernels and multiclass SVMs
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_blobs
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import train_test_split

# 1. The kernel trick: K(x, z) = (x.z + 1)^2 equals phi(x).phi(z) without building phi
x, z = np.array([1.0, 2.0]), np.array([3.0, 0.5])
phi = lambda v: np.array([1, v[0]**2, v[1]**2, np.sqrt(2)*v[0]*v[1],
                          np.sqrt(2)*v[0], np.sqrt(2)*v[1]])
print(f"Kernel (x.z + 1)^2      = {(x @ z + 1) ** 2:.1f}")
print(f"phi(x) . phi(z) (6-dim) = {phi(x) @ phi(z):.1f}   <- same number, less work")

# 2. Linear vs polynomial vs Gaussian (RBF) kernel on two rings
X, y = make_circles(n_samples=400, factor=0.4, noise=0.08, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
print("\nTwo rings (not linearly separable):")
for name, m in (("linear", SVC(kernel="linear")),
                ("polynomial (p=2)", SVC(kernel="poly", degree=2, coef0=1)),
                ("Gaussian / RBF", SVC(kernel="rbf", gamma=1))):
    m.fit(Xtr, ytr)
    print(f"  {name:17s} test accuracy {m.score(Xte, yte):.3f}   support vectors {len(m.support_)}")

# 3. Multiclass with binary SVMs: one-against-the-rest vs one-against-one
X, y = make_blobs(n_samples=600, centers=4, cluster_std=1.8, random_state=4)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
ovr = OneVsRestClassifier(SVC(kernel="rbf")).fit(Xtr, ytr)
ovo = OneVsOneClassifier(SVC(kernel="rbf")).fit(Xtr, ytr)
C = 4
print(f"\n{C} classes:")
print(f"  One-against-the-rest: {len(ovr.estimators_)} SVMs, test accuracy {ovr.score(Xte, yte):.3f}")
print(f"  One-against-one     : {len(ovo.estimators_)} SVMs (C(C-1)/2 = {C*(C-1)//2}), "
      f"test accuracy {ovo.score(Xte, yte):.3f}")
