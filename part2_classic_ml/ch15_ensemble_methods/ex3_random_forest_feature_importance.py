"""
Chapter 15, Example 3: Random Forest Feature Importance
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
One powerful bonus of Random Forest is that it gives you feature importance scores for
free. After training 100 trees, each tree tells us how much each feature contributed to
reducing impurity across all its splits. We average this across all trees to get a
robust importance ranking. On the Breast Cancer dataset (30 features, 569 samples), the
model achieves 97.08% accuracy. The top features (mean concave points, worst concave
points, worst area) are clinically meaningful — they measure tumour shape irregularity.
The bottom features (symmetry error, texture error) contribute almost nothing and could
safely be dropped.

Run it:           python ex3_random_forest_feature_importance.py
Expected output:  expected_output/ex3_random_forest_feature_importance.txt
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
acc = accuracy_score(y_test, rf.predict(X_test))

print(f"Dataset: Breast Cancer | {X.shape[0]} samples, {X.shape[1]} features")
print(f"Test Accuracy: {acc*100:.2f}%\n")
print("Top 10 Most Important Features:")
print(f"  {'Rank':<5} {'Feature':<30} {'Importance':>11}")
print("-" * 50)

ranked = sorted(zip(rf.feature_importances_, data.feature_names), reverse=True)
for rank, (imp, name) in enumerate(ranked[:10], 1):
    bar = 'X' * int(imp * 150)
    print(f"  {rank:<5} {name:<30} {imp:>10.4f}  {bar}")

print()
print("Bottom 5 (candidates to drop):")
for imp, name in ranked[-5:]:
    print(f"  {name:<30}: {imp:.4f}")
