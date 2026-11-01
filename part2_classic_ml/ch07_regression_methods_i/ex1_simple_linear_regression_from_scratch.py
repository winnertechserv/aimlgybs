"""
Chapter 7, Example 1: Simple Linear Regression from Scratch
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This program uses the exact closed-form formulas on the salaries of 10 employees. It
finds the averages (9.0 years, 59.9 thousand), then the slope 3.04 and intercept 32.52.
The table shows each training point, the line's prediction and the error. Finally it
tests a new person with 10 years' experience: predicted 62.94, actual 64, squared error
1.12.

Run it:           python ex1_simple_linear_regression_from_scratch.py
Expected output:  expected_output/ex1_simple_linear_regression_from_scratch.txt
"""
# Chapter 7 - Example 1: Simple linear regression from scratch (method of least squares)
# Salary (in $1000s) vs years of experience for 10 employees.
import numpy as np

x = np.array([1, 3, 4, 6, 7, 9, 11, 13, 16, 20], dtype=float)
y = np.array([35, 40, 47, 50, 56, 58, 67, 71, 82, 93], dtype=float)
N = len(x)

mu_x, mu_y = x.mean(), y.mean()
# Closed-form least-squares solution (set both derivatives of the error to zero)
w = np.sum((x - mu_x) * (y - mu_y)) / np.sum((x - mu_x) ** 2)
w0 = mu_y - w * mu_x
print(f"Mean of x = {mu_x:.1f}, mean of y = {mu_y:.1f}")
print(f"Slope w   = {w:.2f}   (each extra year adds about ${w * 1000:,.0f})")
print(f"Intercept w0 = {w0:.2f}  (predicted starting salary)")

# Training error
y_hat = w * x + w0
E = 0.5 * np.sum((y_hat - y) ** 2)
print(f"\nSum-of-squares error on training data, E = {E:.2f}")
print("  x    y   predicted  error")
for xi, yi, pi in zip(x, y, y_hat):
    print(f" {xi:3.0f}  {yi:3.0f}   {pi:7.2f}  {pi - yi:6.2f}")

# Test phase: a new person with 10 years of experience earns 64 (thousand)
x_test, y_test = 10, 64.0
pred = w * x_test + w0
print(f"\nTest: 10 years -> predicted {pred:.2f}, actual {y_test:.0f}, "
      f"squared error {(pred - y_test) ** 2:.2f}")
