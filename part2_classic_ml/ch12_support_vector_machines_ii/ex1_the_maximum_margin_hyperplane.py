"""
Chapter 12, Example 1: The Maximum-Margin Hyperplane
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This uses the same data as the perceptron in Chapter 11, now with a linear SVM with a
huge C (effectively a hard margin). Only 2 of the 40 points become support vectors. Both
have y·g(x) = 1.000, which means they sit exactly on the margin. Rebuilding w from those
two points alone, as Σ αₙyₙxₙ, gives the same w. The last line is the payoff: the
closest point is 0.351 units from the SVM boundary, against only 0.032 for the
perceptron's line. That is more than ten times the safety gap.

Run it:           python ex1_the_maximum_margin_hyperplane.py
Expected output:  expected_output/ex1_the_maximum_margin_hyperplane.txt
"""
# Chapter 12 - Example 1: Maximum-margin hyperplane (hard-margin linear SVM)
# Same data as the perceptron example, so we can compare the two lines.
import numpy as np
from sklearn.svm import SVC

rng = np.random.default_rng(1)
X = np.vstack([rng.normal([1, 1], 0.5, (20, 2)), rng.normal([-1, -1], 0.5, (20, 2))])
y = np.r_[np.ones(20), -np.ones(20)]

svm = SVC(kernel="linear", C=1e6).fit(X, y)          # huge C = (almost) hard margin
w, w0 = svm.coef_[0], svm.intercept_[0]
print(f"SVM boundary: {w[0]:.3f}*x1 + {w[1]:.3f}*x2 + {w0:.3f} = 0")
print(f"Number of support vectors: {len(svm.support_)} (out of {len(X)} points)")
for sv, a in zip(svm.support_vectors_, svm.dual_coef_[0]):
    g = sv @ w + w0
    print(f"  support vector {np.round(sv, 3)}  y*g(x) = {np.sign(a) * g:.3f}  "
          f"alpha*y = {a:+.3f}")
print(f"\nMargin (distance from boundary to nearest point) = 1/||w|| = {1 / np.linalg.norm(w):.3f}")
print(f"Width of the empty 'street' between the classes  = 2/||w|| = {2 / np.linalg.norm(w):.3f}")

# Rebuild w from the support vectors only: w = sum(alpha_n * y_n * x_n)
w_from_sv = (svm.dual_coef_[0][:, None] * svm.support_vectors_).sum(axis=0)
print("w rebuilt from support vectors:", np.round(w_from_sv, 3), "(matches w)")

# Compare with the perceptron line learned in Chapter 11 on this exact data
w_perc = np.array([0.9, 0.066, 1.635])              # [w0, w1, w2] from Chapter 11, Example 2
d_perc = np.abs(np.column_stack([np.ones(40), X]) @ w_perc).min() / np.linalg.norm(w_perc[1:])
print(f"\nClosest point to the PERCEPTRON line: {d_perc:.3f}  vs  SVM: {1 / np.linalg.norm(w):.3f}")
