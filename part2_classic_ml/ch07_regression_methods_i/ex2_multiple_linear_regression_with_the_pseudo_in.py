"""
Chapter 7, Example 2: Multiple Linear Regression with the Pseudo-Inverse
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Two inputs (humidity and pressure) predict temperature. We build the data matrix with a
column of 1s and solve w = (XᵀX)⁻¹Xᵀy. The normal equation and NumPy's pseudo-inverse
give the same weights: [178.60, −0.2754, −0.1349]. The test prediction is 22.84 °C
against an actual 21.7 °C. Across the training days the RMSE is 1.21 °C, under 5% of the
average temperature.

Run it:           python ex2_multiple_linear_regression_with_the_pseudo_in.py
Expected output:  expected_output/ex2_multiple_linear_regression_with_the_pseudo_in.txt
"""
# Chapter 7 - Example 2: Multiple linear regression with the pseudo-inverse
# Predict temperature (C) from humidity (%) and air pressure (hPa) on 15 days.
import numpy as np
np.set_printoptions(suppress=True)

data = np.array([
    [45.3, 1012.7, 29.3], [46.0, 1017.9, 29.3], [53.7, 1021.4, 25.2],
    [62.7, 1023.2, 22.0], [81.8, 1023.1, 18.8], [59.8, 1021.6, 23.4],
    [57.2, 1018.5, 23.7], [82.4, 1022.0, 18.6], [81.5, 1028.3, 18.7],
    [43.5, 1011.0, 32.0], [48.4, 1024.5, 27.1], [68.5, 1027.9, 21.3],
    [55.5, 1014.5, 24.1], [42.9, 1016.8, 31.6], [42.8, 1012.3, 30.6]])
X_raw, y = data[:, :2], data[:, 2]

# Data matrix: a column of 1s (for w0) followed by the inputs
X = np.column_stack([np.ones(len(X_raw)), X_raw])
print("Data matrix X has shape", X.shape, "(N rows, d+1 columns)")

w = np.linalg.inv(X.T @ X) @ X.T @ y        # w = (X^T X)^-1 X^T y
w_pinv = np.linalg.pinv(X) @ y              # same thing, numerically safer
print("w (normal equation) =", np.round(w, 4))
print("w (pseudo-inverse)  =", np.round(w_pinv, 4))

x_test = np.array([1, 66.0, 1020.0])
pred = x_test @ w
print(f"\nTest: humidity 66.0, pressure 1020.0 -> predicted {pred:.2f} C "
      f"(actual 21.7, squared error {(pred - 21.7) ** 2:.4f})")

y_hat = X @ w
rmse = np.sqrt(np.mean((y_hat - y) ** 2))
print(f"Training RMSE = {rmse:.3f} C  ({100 * rmse / y.mean():.1f}% of the mean temperature)")
print("Meaning of the weights: each extra 1% humidity changes temperature by "
      f"{w[1]:+.3f} C, holding pressure fixed.")
