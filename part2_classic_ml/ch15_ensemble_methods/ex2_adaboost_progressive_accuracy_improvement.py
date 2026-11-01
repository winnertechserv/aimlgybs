"""
Chapter 15, Example 2: AdaBoost — Progressive Accuracy Improvement
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
AdaBoost is sequential Bagging's smarter cousin. Instead of building classifiers in
parallel on random subsets, it builds them one after another, always paying extra
attention to the examples that were misclassified so far. We track how the ensemble
accuracy grows as we add more weak classifiers. A single stump starts at 70%. By round
50 it reaches 85% — a 15 percentage point gain purely from sequentially reweighting
examples. The final comparison shows AdaBoost vs Random Forest vs a single tree: the
random forest clearly beats the single tree, while AdaBoost with 200 stumps only matches
it, because its accuracy dips slightly after round 50. More rounds are not always
better.

Run it:           python ex2_adaboost_progressive_accuracy_improvement.py
Expected output:  expected_output/ex2_adaboost_progressive_accuracy_improvement.txt
"""
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(0)
X, y = make_classification(n_samples=400, n_features=10, n_informative=6,
                            n_classes=2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("=== AdaBoost: Accuracy vs Number of Weak Classifiers ===\n")
print(f"{'Classifiers':>12}  {'Ensemble Acc':>14}  {'vs Baseline'}")
print("-" * 45)

single_acc = None
for n in [1, 2, 5, 10, 20, 50, 100, 200]:
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                             n_estimators=n, random_state=0)
    ada.fit(X_train, y_train)
    acc = accuracy_score(y_test, ada.predict(X_test))
    if n == 1: single_acc = acc
    diff = f"+{(acc-single_acc)*100:.1f}%" if n > 1 else "(baseline)"
    print(f"{n:>12}  {acc*100:>13.1f}%  {diff}")

print(f"\nBaseline (1 stump): {single_acc*100:.1f}%")
print()
print("=== Final Model Comparison ===\n")
for name, model in [
    ('Single Tree (depth=5)',    DecisionTreeClassifier(max_depth=5, random_state=0)),
    ('Random Forest 100 trees',  RandomForestClassifier(n_estimators=100, random_state=0)),
    ('AdaBoost 200 stumps',      AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                                     n_estimators=200, random_state=0)),
]:
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  {name:<28}: {acc*100:.1f}%")
