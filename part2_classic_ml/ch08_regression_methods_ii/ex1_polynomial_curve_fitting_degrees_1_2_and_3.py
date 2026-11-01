"""
Chapter 8, Example 1: Polynomial Curve Fitting (Degrees 1, 2 and 3)
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Humidity is predicted from temperature with polynomials of degree 1, 2 and 3, and each
model is tested on five days it never saw. np.vander builds the design matrix (columns
1, x, x², ...) and the pseudo-inverse solves for the weights. The straight line misses
the curve (test RMSE 3.56). Degree 2 follows it closely (0.60). Degree 3 fits the
training days slightly better (0.76 against 0.84) but does worse on the new days (0.86).
That is an early sign of overfitting, which Example 2 explores fully.

Run it:           python ex1_polynomial_curve_fitting_degrees_1_2_and_3.py
Expected output:  expected_output/ex1_polynomial_curve_fitting_degrees_1_2_and_3.txt
"""
# Chapter 8 - Example 1: Polynomial curve fitting - humidity from temperature
# Degree M = 1, 2, 3 using the design (Vandermonde) matrix and the pseudo-inverse.
import numpy as np
np.set_printoptions(suppress=True)

temp = np.array([29.3, 29.3, 25.2, 22.0, 18.8, 23.4, 23.7, 18.6, 18.7, 32.0,
                 27.1, 21.3, 24.1, 31.6, 30.6])
hum = np.array([45.3, 46.0, 53.7, 62.7, 81.8, 59.8, 57.2, 82.4, 81.5, 43.5,
                48.4, 68.5, 55.5, 42.9, 42.8])
# Five new days the model never saw during training
t_test = np.array([19.5, 22.8, 26.0, 28.4, 31.2])
h_test = np.array([77.5, 61.4, 50.5, 44.8, 42.8])

for M in (1, 2, 3):
    Phi = np.vander(temp, M + 1, increasing=True)      # columns: 1, x, x^2, ..., x^M
    w = np.linalg.pinv(Phi) @ hum                      # w = (Phi^T Phi)^-1 Phi^T y
    train_rmse = np.sqrt(np.mean((Phi @ w - hum) ** 2))
    pred = np.vander(t_test, M + 1, increasing=True) @ w
    test_rmse = np.sqrt(np.mean((pred - h_test) ** 2))
    print(f"M = {M}: Phi is {Phi.shape[0]}x{Phi.shape[1]}, w = {np.round(w, 4)}")
    print(f"       training RMSE = {train_rmse:.2f}   test RMSE (5 new days) = {test_rmse:.2f}")
    print(f"       predictions {np.round(pred, 1)}  vs actual {h_test}\n")
print("M = 2 fits the curve; M = 3 adds flexibility the data does not need.")
