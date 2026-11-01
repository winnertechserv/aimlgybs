"""
Chapter 11, Example 3: Where the Perceptron Fails
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
The XOR pattern puts one class on two opposite corners and the other class on the other
two. The perceptron keeps making mistakes epoch after epoch and never converges. Scikit-
learn's perceptron scores only 42%, worse than guessing. K-NN, which learns a nonlinear
boundary, scores 100%. With overlapping classes the perceptron also never reaches zero
errors. These are exactly the gaps that SVMs (and later neural networks) fill.

Run it:           python ex3_where_the_perceptron_fails.py
Expected output:  expected_output/ex3_where_the_perceptron_fails.txt
"""
# Chapter 11 - Example 3: Where the perceptron fails - data that a line cannot separate
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(0)
# XOR pattern: opposite corners belong to the same class
centers = [([0, 0], -1), ([1, 1], -1), ([0, 1], 1), ([1, 0], 1)]
X = np.vstack([rng.normal(c, 0.1, (25, 2)) for c, _ in centers])
y = np.repeat([lab for _, lab in centers], 25)

def train_perceptron(X, y, epochs=50, eta=0.1):
    Xh = np.column_stack([np.ones(len(X)), X]); w = np.zeros(3); history = []
    for _ in range(epochs):
        errs = 0
        for xn, yn in zip(Xh, y):
            if yn * (w @ xn) <= 0:
                w += eta * yn * xn; errs += 1
        history.append(errs)
        if errs == 0: break
    return w, history

w, hist = train_perceptron(X, y)
print("XOR data - mistakes per epoch (first 10):", hist[:10])
print(f"After {len(hist)} epochs still {hist[-1]} mistakes -> it never converges.")
print(f"Perceptron (scikit-learn) accuracy on XOR: "
      f"{Perceptron(random_state=0).fit(X, y).score(X, y):.2f}")

# A non-linear method handles it easily (K-NN learns a curved boundary implicitly)
print(f"K-NN (K=5) accuracy on XOR: {KNeighborsClassifier(5).fit(X, y).score(X, y):.2f}")

# Overlapping classes: even the best line makes some mistakes
Xo = np.vstack([rng.normal([0, 0], 1, (100, 2)), rng.normal([1.5, 1.5], 1, (100, 2))])
yo = np.r_[-np.ones(100), np.ones(100)]
w, hist = train_perceptron(Xo, yo)
print(f"\nOverlapping data: mistakes in the last 5 epochs {hist[-5:]} - never reaches 0.")
