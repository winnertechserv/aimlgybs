"""
Chapter 11, Example 1: Discriminant Functions and Bayes Decision Boundaries
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
The first part evaluates g(x) = x₁ + x₂ − 1 at a few example points and reports which
side each one is on. The second part checks the three-case theory. When two classes
share a covariance matrix, a linear boundary (LDA) does just as well as a quadratic one
(QDA), and the quadratic term W is almost zero. When the covariances differ, one class
is tight and the other spread out, so the quadratic boundary wins (93.0% vs 87.5%) and W
is clearly non-zero, which means the boundary curves.

Run it:           python ex1_discriminant_functions_and_bayes_decision_bou.py
Expected output:  expected_output/ex1_discriminant_functions_and_bayes_decision_bou.txt
"""
# Chapter 11 - Example 1: Discriminant functions - which side of the line is a point on?
# And how the covariance decides whether a Bayes boundary is straight or curved.
import numpy as np
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)

# 1. The line x1 + x2 - 1 = 0 has discriminant g(x) = x1 + x2 - 1
g = lambda x: x[0] + x[1] - 1
for p in [(1, 0), (0, 1), (0, 0), (1, 1), (0.2, 0.3)]:
    v = g(p)
    side = "on the line" if v == 0 else "positive side" if v > 0 else "negative side"
    print(f"g{p} = {v:+.1f}  -> {side}")

# 2. Bayes classifier with Gaussian classes
rng = np.random.default_rng(0)
def sample(mean, cov, n=300): return rng.multivariate_normal(mean, cov, n)

cases = {
    "Same covariance (Case 2)": ([[2, 1], [1, 2]], [[2, 1], [1, 2]]),
    "Different covariance (Case 3)": ([[0.3, 0], [0, 0.3]], [[4, 0], [0, 4]]),
}
for name, (c0, c1) in cases.items():
    X = np.vstack([sample([0, 0], c0), sample([2, 2], c1)])
    y = np.r_[np.zeros(300), np.ones(300)]
    lda = LinearDiscriminantAnalysis().fit(X, y)       # assumes a shared covariance -> line
    qda = QuadraticDiscriminantAnalysis().fit(X, y)    # one covariance per class -> curve
    print(f"\n{name}")
    print(f"  Linear boundary accuracy   : {lda.score(X, y):.3f}")
    print(f"  Quadratic boundary accuracy: {qda.score(X, y):.3f}")
    # Size of the x^T W x term: W = inverse(cov1) - inverse(cov0)
    W = np.linalg.inv(np.cov(X[y == 1].T)) - np.linalg.inv(np.cov(X[y == 0].T))
    print(f"  Size of quadratic term |W| = {np.abs(W).max():.2f}"
          f"  -> boundary is {'almost straight' if np.abs(W).max() < 0.3 else 'curved'}")
