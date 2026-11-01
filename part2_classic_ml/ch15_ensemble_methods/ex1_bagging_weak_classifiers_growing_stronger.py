"""
Chapter 15, Example 1: Bagging — Weak Classifiers Growing Stronger
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This example shows the magic of Bagging: taking individually weak shallow trees (depth
2) and combining 10 to 200 of them into a stronger ensemble. A single shallow tree gets
75.3%, but 50 bagged ones push to 84.7%. The Random Forest takes this further by also
randomizing which features each tree considers at each split — it achieves 91.3%. Notice
also that a single deep (unlimited) tree gets 84.0% — similar to the bagged shallow
trees but with more risk of overfitting. The visual bar chart makes the improvement
instantly obvious.

Run it:           python ex1_bagging_weak_classifiers_growing_stronger.py
Expected output:  expected_output/ex1_bagging_weak_classifiers_growing_stronger.txt
"""
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                            n_redundant=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"{'Model':<38} {'Test Acc':>10}")
print("-" * 50)
results = {}

# Single trees
for name, clf in [('Single Tree (deep)', DecisionTreeClassifier(random_state=42)),
                   ('Single Shallow Tree (depth=2)', DecisionTreeClassifier(max_depth=2, random_state=42))]:
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    results[name] = acc

# Bagging
for L in [10, 50, 100, 200]:
    bag = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=2),
                            n_estimators=L, random_state=42, bootstrap=True)
    bag.fit(X_train, y_train)
    results[f'Bagging {L:>3} shallow trees'] = accuracy_score(y_test, bag.predict(X_test))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
results['Random Forest (100 trees)'] = accuracy_score(y_test, rf.predict(X_test))

for name, acc in results.items():
    bar = 'X' * int(acc * 40)
    print(f"  {name:<36} {acc*100:>6.1f}%  |{bar}")

print()
print("Key insight: Weak classifiers (shallow trees) become strong when bagged together.")
print("Random Forest adds feature randomness on top of Bagging for even better results.")
