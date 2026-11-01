"""
Chapter 8, Example 2: Watching Overfitting Happen
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We take 10 noisy points from a sine wave and fit polynomials from degree 0 to 9. Read
the table from top to bottom. Training error keeps falling, all the way to 0 at M = 9,
but test error is lowest around M = 3 and then climbs again. The largest weight explodes
to about 95,000, which is the warning sign. The second table fixes the degree at 9 and
adds data: with 15 or 100 points the test error drops back to about 0.32.

Run it:           python ex2_watching_overfitting_happen.py
Expected output:  expected_output/ex2_watching_overfitting_happen.txt
"""
# Chapter 8 - Example 2: Model complexity vs overfitting
# Fit polynomials of degree M to 10 noisy points from a sine wave, then test on new points.
import numpy as np

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)            # the true (hidden) curve

def make(n, even=True):
    x = np.linspace(0, 1, n) if even else rng.uniform(0, 1, n)   # training x evenly spaced
    return x, f(x) + rng.normal(0, 0.3, n)

def fit_rmse(x_tr, y_tr, x_te, y_te, M):
    Phi = np.vander(x_tr, M + 1, increasing=True)
    w = np.linalg.lstsq(Phi, y_tr, rcond=None)[0]
    r = lambda x, y: np.sqrt(np.mean((np.vander(x, M + 1, increasing=True) @ w - y) ** 2))
    return r(x_tr, y_tr), r(x_te, y_te), np.abs(w).max()

x_te, y_te = make(500, even=False)
x10, y10 = make(10)
print("N = 10 training points")
print(" M   train RMSE   test RMSE   largest |w|")
for M in (0, 1, 3, 6, 9):
    tr, te, wmax = fit_rmse(x10, y10, x_te, y_te, M)
    print(f" {M}     {tr:.3f}       {te:8.3f}   {wmax:12.1f}")
print("M = 9 passes through all 10 points (train error ~0) but is much worse on new data.")

print("\nMore data tames the same complex model (M = 9):")
for n in (10, 15, 100):
    x, y = (x10, y10) if n == 10 else make(n)
    tr, te, _ = fit_rmse(x, y, x_te, y_te, 9)
    print(f"  N = {n:3d}: train RMSE {tr:.3f}, test RMSE {te:8.3f}")
