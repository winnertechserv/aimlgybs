"""
Chapter 11, Example 2: The Perceptron Learning Algorithm
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
The perceptron algorithm, line by line. We start from a deliberately bad line so that
the first epoch has plenty of mistakes (11). The weights are nudged after each one, and
by epoch 2 there are no mistakes, so training stops. Then we classify three new points
by the sign of g(x). Look at the last line: the closest training point is only 0.032
units from the boundary. The perceptron stopped at the first line that worked, not the
safest one. Keep that number in mind for Chapter 12.

Run it:           python ex2_the_perceptron_learning_algorithm.py
Expected output:  expected_output/ex2_the_perceptron_learning_algorithm.txt
"""
# Chapter 11 - Example 2: The perceptron learning algorithm from scratch
import numpy as np

rng = np.random.default_rng(1)
X = np.vstack([rng.normal([1, 1], 0.5, (20, 2)), rng.normal([-1, -1], 0.5, (20, 2))])
y = np.r_[np.ones(20), -np.ones(20)]                 # labels +1 and -1
Xh = np.column_stack([np.ones(len(X)), X])           # x_hat = [1, x1, x2] so w0 is the bias

w = np.array([2.0, -1.0, 0.5])                       # 1. start from a deliberately bad line
eta = 0.1
for epoch in range(1, 101):
    mistakes = 0
    for xn, yn in zip(Xh, y):                        # 2. present each example
        s = w @ xn
        if yn == 1 and s <= 0:                       # 3. wrong side -> nudge w
            w += eta * xn; mistakes += 1
        elif yn == -1 and s > 0:
            w -= eta * xn; mistakes += 1
    print(f"Epoch {epoch}: misclassifications = {mistakes}")
    if mistakes == 0:                                # 5. stop when nothing is wrong
        break

print(f"\nLearned w = [w0, w1, w2] = {np.round(w, 3)}")
print(f"Boundary: {w[1]:.2f}*x1 + {w[2]:.2f}*x2 + {w[0]:.2f} = 0")
for t in ([1.5, 1.0], [-0.5, -1.5], [0.5, 0.4]):
    s = w @ np.r_[1, t]
    print(f"Test {t}: g(x) = {s:+.2f} -> class {'+1' if s > 0 else '-1'}")

# Distance of the closest training point to the line (the margin this perceptron got)
dist = np.abs(Xh @ w) / np.linalg.norm(w[1:])
print(f"Closest training point is {dist.min():.3f} units from the boundary")
