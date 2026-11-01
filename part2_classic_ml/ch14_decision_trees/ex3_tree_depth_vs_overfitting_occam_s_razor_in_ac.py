"""
Chapter 14, Example 3: Tree Depth vs Overfitting (Occam's Razor in Action)
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This example shows Occam's Razor in decision trees — the principle that simpler models
are preferred when they explain the data equally well. We train the same tree on the
Iris dataset with different maximum depths and compare training vs test accuracy. A very
deep tree achieves 100% training accuracy (it perfectly memorizes the data) but test
accuracy may not improve and can drop. The sweet spot is around depth 2-3 for Iris. We
also compare Gini vs Entropy criteria, confirming they produce nearly identical results
in practice — the choice rarely matters much.

Run it:           python ex3_tree_depth_vs_overfitting_occam_s_razor_in_ac.py
Expected output:  expected_output/ex3_tree_depth_vs_overfitting_occam_s_razor_in_ac.txt
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("=== Tree Depth vs Accuracy (Iris Dataset) ===\n")
print(f"{'Max Depth':<12} {'Train Acc':>10} {'Test Acc':>10} {'Leaf Nodes':>12}")
print("-" * 48)
for depth in [1, 2, 3, 4, 5, None]:
    clf = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=42)
    clf.fit(X_train, y_train)
    tr = accuracy_score(y_train, clf.predict(X_train))
    te = accuracy_score(y_test,  clf.predict(X_test))
    d  = str(depth) if depth else "No limit"
    print(f"  {d:<10} {tr*100:>9.1f}% {te*100:>9.1f}% {clf.get_n_leaves():>12}")

print()
print("Insight: Deeper trees memorize training data (Occam's Razor says stop early!)")
print()
print("=== Gini vs Entropy Impurity Criterion ===\n")
for crit in ['gini', 'entropy']:
    clf = DecisionTreeClassifier(max_depth=3, criterion=crit, random_state=42)
    clf.fit(X_train, y_train)
    te = accuracy_score(y_test, clf.predict(X_test))
    print(f"  criterion='{crit}': Test={te*100:.1f}%  Leaves={clf.get_n_leaves()}")
print("\nConclusion: Gini and Entropy usually produce very similar trees in practice.")
