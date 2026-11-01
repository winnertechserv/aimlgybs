"""
Chapter 19, Example 2: L1 vs L2 Regularization — Effect on Weights
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We train a logistic regression model on a 20-feature dataset where only 5 features are
truly informative. Without regularization, all 20 weights are non-zero — the model uses
noise features. L2 regularization shrinks all weights toward zero proportionally. L1
regularization does something smarter: it pushes weights to exactly zero (sparse
solution). With strong L1 (C=0.1), 15 features get zeroed out — the model correctly
identifies the informative features! This is the key practical difference: L2 shrinks,
L1 selects.

Run it:           python ex2_l1_vs_l2_regularization_effect_on_weights.py
Expected output:  expected_output/ex2_l1_vs_l2_regularization_effect_on_weights.txt
"""
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=20, n_informative=5,
                            n_redundant=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)

print("=== L1 vs L2 Regularization ===\n")
print("Dataset: 20 features, only 5 truly informative\n")
print(f"{'Regularization':<22} {'Test Acc':>9} {'||w||':>8} {'Zero weights':>14}")
print("-" * 58)
results = {}
for name, C, l1r in [
    ('No Reg (C=1e6)',  1e6, 0.0),
    ('L2 (C=1.0)',      1.0, 0.0),
    ('L2 strong (C=0.1)', 0.1, 0.0),
    ('L1 (C=1.0)',      1.0, 1.0),
    ('L1 strong (C=0.1)', 0.1, 1.0),
]:
    clf = LogisticRegression(C=C, l1_ratio=l1r, solver='saga',
                             max_iter=2000, random_state=42)
    clf.fit(X_tr, y_tr)
    acc  = accuracy_score(y_te, clf.predict(X_te))
    w    = clf.coef_[0]
    n_z  = np.sum(np.abs(w) < 1e-4)
    wn   = np.linalg.norm(w)
    results[name] = w
    print(f"  {name:<20} {acc*100:>8.1f}% {wn:>8.3f} {n_z:>14}")

print("\n=== No-Reg vs L1-Strong: Per-Feature Weight Comparison ===\n")
w_none = results['No Reg (C=1e6)']
w_l1   = results['L1 strong (C=0.1)']
print(f"  {'Feature':>10} {'No Reg':>10} {'L1 Strong':>12}  Status")
print("-" * 48)
for i, (wn, wl) in enumerate(zip(w_none, w_l1)):
    status = "ZEROED OUT" if abs(wl) < 1e-4 else "Active"
    print(f"  Feature {i+1:>2}  {wn:>10.4f} {wl:>12.4f}  {status}")
