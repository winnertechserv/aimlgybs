"""
Chapter 2, Example 3: Fraud Detection in Banking
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Banks see millions of normal transactions and very few frauds, often with no fraud
labels at all. An Isolation Forest learns what normal looks like and flags anything
unusual. Here, 20 frauds (large amounts, at night, far from home, many purchases in an
hour) are hidden among 2,000 normal transactions. The model flags 21 transactions and
catches all 20 frauds, with only one false alarm. We use the true labels only at the
end, to check the result.

Run it:           python ex3_fraud_detection_in_banking.py
Expected output:  expected_output/ex3_fraud_detection_in_banking.txt
"""
# Chapter 2 - Example 3: AI in banking - spotting fraudulent transactions (anomaly detection)
# No fraud labels are used for training: Isolation Forest learns what 'normal' looks like.
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

rng = np.random.default_rng(0)
n_normal, n_fraud = 2000, 20
# Features: amount ($), hour of day, distance from home (km), purchases in last hour
normal = np.column_stack([rng.lognormal(3.5, 0.6, n_normal), rng.normal(14, 3, n_normal) % 24,
                          rng.exponential(5, n_normal), rng.poisson(1, n_normal)])
fraud = np.column_stack([rng.lognormal(6.5, 0.5, n_fraud), rng.normal(3, 1, n_fraud) % 24,
                         rng.exponential(400, n_fraud), rng.poisson(6, n_fraud)])
X = np.vstack([normal, fraud])
y = np.r_[np.zeros(n_normal), np.ones(n_fraud)]          # used ONLY to check the result

model = IsolationForest(n_estimators=200, contamination=0.01, random_state=0).fit(X)
flag = (model.predict(X) == -1).astype(int)              # -1 means 'anomaly'
score = -model.score_samples(X)                          # higher = more unusual

print(f"Transactions: {len(X)}  (true frauds hidden inside: {n_fraud})")
print(f"Flagged as suspicious: {flag.sum()}")
print(f"Precision: {precision_score(y, flag):.2f}   Recall: {recall_score(y, flag):.2f}")

print("\nTop 5 most unusual transactions:")
print("  amount($)  hour  distance(km)  buys/hr  truly fraud?")
for i in np.argsort(score)[::-1][:5]:
    a, h, d, b = X[i]
    print(f"  {a:9.0f}  {h:4.0f}  {d:12.0f}  {b:7.0f}  {'yes' if y[i] else 'no'}")
