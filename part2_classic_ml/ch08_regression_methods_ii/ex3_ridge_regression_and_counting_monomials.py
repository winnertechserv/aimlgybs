"""
Chapter 8, Example 3: Ridge Regression and Counting Monomials
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We keep the overfitting degree-9 model and add an L2 penalty λ. With λ = 0 the test RMSE
is 0.97 and the weights are in the millions. A tiny λ of 0.0001 cuts the test error to
0.35 and the weight size to about 20. A large λ of 1 overshoots and underfits. The
second part checks the formula D = (d+M)!/(d!M!), including the 364 terms for 11 inputs
at degree 3, and builds φ([2, 10]).

Run it:           python ex3_ridge_regression_and_counting_monomials.py
Expected output:  expected_output/ex3_ridge_regression_and_counting_monomials.txt
"""
# Chapter 8 - Example 3: Ridge regression (L2 regularisation) and polynomial surfaces
import numpy as np
from math import factorial

rng = np.random.default_rng(2)
f = lambda x: np.sin(2 * np.pi * x)
x_tr = np.linspace(0, 1, 10); y_tr = f(x_tr) + rng.normal(0, 0.3, 10)
x_te = rng.uniform(0, 1, 500); y_te = f(x_te) + rng.normal(0, 0.3, 500)

M = 9
Phi = np.vander(x_tr, M + 1, increasing=True)
Phi_te = np.vander(x_te, M + 1, increasing=True)
print("Degree M = 9, N = 10 points. Ridge: w = (Phi^T Phi + lambda I)^-1 Phi^T y")
print(" lambda      train RMSE  test RMSE  size of w (||w||)")
for lam in (0, 1e-6, 1e-4, 1e-2, 1):
    if lam == 0:
        w = np.linalg.lstsq(Phi, y_tr, rcond=None)[0]       # plain least squares
    else:
        w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(M + 1), Phi.T @ y_tr)
    tr = np.sqrt(np.mean((Phi @ w - y_tr) ** 2))
    te = np.sqrt(np.mean((Phi_te @ w - y_te) ** 2))
    print(f" {lam:<9g}   {tr:.3f}      {te:8.3f}   {np.linalg.norm(w):12.1f}")
print("A small lambda shrinks the weights and fixes overfitting; too large underfits.")

# How many terms does a polynomial in d inputs of degree M have?  D = (d+M)! / (d! M!)
print("\nNumber of monomials D = (d+M)!/(d! M!):")
for d, Mdeg in ((2, 2), (2, 3), (3, 2), (11, 3)):
    D = factorial(d + Mdeg) // (factorial(d) * factorial(Mdeg))
    print(f"  d = {d:2d} inputs, degree M = {Mdeg}: D = {D}")

# Monomials for x = [2, 10], M = 2: 1, x1, x2, x1^2, x2^2, x1*x2
x1, x2 = 2, 10
print("\nphi([2, 10]) for M = 2:", [1, x1, x2, x1**2, x2**2, x1 * x2])
