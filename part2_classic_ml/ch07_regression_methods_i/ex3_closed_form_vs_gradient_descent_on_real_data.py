"""
Chapter 7, Example 3: Closed Form vs Gradient Descent on Real Data
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Scikit-learn's diabetes dataset has 442 patients and 10 measurements each. We fit linear
regression two ways. The closed form gives the answer in one step. Gradient descent
starts from zero and walks downhill: watch the training error fall from 23,375 to 2,804.
Both reach the same test RMSE (about 55.7), and so does scikit-learn. We standardize the
features first because gradient descent converges much faster when all inputs are on the
same scale.

Run it:           python ex3_closed_form_vs_gradient_descent_on_real_data.py
Expected output:  expected_output/ex3_closed_form_vs_gradient_descent_on_real_data.txt
"""
# Chapter 7 - Example 3: Closed form vs gradient descent, judged on unseen test data
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X, y = load_diabetes(return_X_y=True)        # 442 patients, 10 measurements each
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler().fit(X_tr)           # same scale for every feature helps GD
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
add1 = lambda A: np.column_stack([np.ones(len(A)), A])

# 1. Closed form (normal equation via the pseudo-inverse)
w_cf = np.linalg.pinv(add1(X_tr)) @ y_tr

# 2. Gradient descent: walk downhill on the error surface step by step
A = add1(X_tr)
w_gd = np.zeros(A.shape[1])
lr = 0.1
for epoch in range(1, 3001):
    grad = A.T @ (A @ w_gd - y_tr) / len(y_tr)
    w_gd -= lr * grad
    if epoch in (1, 10, 100, 1000, 3000):
        mse = np.mean((A @ w_gd - y_tr) ** 2)
        print(f"GD epoch {epoch:5d}: training MSE = {mse:9.1f}")

# 3. scikit-learn for comparison
sk = LinearRegression().fit(X_tr, y_tr)

def rmse(pred): return np.sqrt(np.mean((pred - y_te) ** 2))
print(f"\nTest RMSE  closed form     : {rmse(add1(X_te) @ w_cf):.2f}")
print(f"Test RMSE  gradient descent: {rmse(add1(X_te) @ w_gd):.2f}")
print(f"Test RMSE  scikit-learn    : {rmse(sk.predict(X_te)):.2f}")
print(f"Largest weight difference, closed form vs GD: {np.abs(w_cf - w_gd).max():.3f}")
print(f"Test RMSE as % of the mean target: {100 * rmse(add1(X_te) @ w_cf) / y_te.mean():.1f}%")
