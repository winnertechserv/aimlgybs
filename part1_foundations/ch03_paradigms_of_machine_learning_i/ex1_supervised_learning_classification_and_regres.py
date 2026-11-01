"""
Chapter 3, Example 1: Supervised Learning — Classification and Regression
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Three short supervised tasks, each learned from labelled data. First, a classifier
learns to tell children from adults using 20 example people, then labels two new people.
Second, a K-NN classifier learns the three Iris species and is tested on flowers it
never saw: the test accuracy is what measures generalisation. Third, a regression line
predicts salary from experience. The answer, salary = 3.04 x years + 32.52, is the same
line you will derive by hand in Chapter 7.

Run it:           python ex1_supervised_learning_classification_and_regres.py
Expected output:  expected_output/ex1_supervised_learning_classification_and_regres.txt
"""
# Chapter 3 - Example 1: Supervised learning - classification AND regression
# Both learn a function y = f(x) from LABELLED examples, then predict for new x.
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# --- 1. Classification: child (0) or adult (1) from height (cm) and weight (kg)
X = np.array([[98, 16.4], [104, 15.9], [110, 19.8], [115, 19.1], [119, 23.0],
              [123, 22.4], [127, 27.1], [131, 26.3], [134, 31.0], [101, 17.5],
              [152, 52.0], [158, 50.5], [161, 63.2], [165, 57.8], [168, 71.0],
              [172, 64.5], [175, 80.3], [179, 72.9], [183, 88.0], [188, 81.4]])
y = np.r_[np.zeros(10, int), np.ones(10, int)]
clf = LogisticRegression(max_iter=1000).fit(X, y)
for person in ([150, 47.0], [110, 20.0]):
    p = clf.predict_proba([person])[0, 1]
    print(f"Height {person[0]} cm, weight {person[1]} kg -> "
          f"{'Adult' if p >= 0.5 else 'Child'} (P(adult) = {p:.2f})")

# --- 2. Multi-class classification: the Iris flowers, with a proper train/test split
iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.3,
                                      stratify=iris.target, random_state=1)
knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
print(f"\nIris: trained on {len(ytr)} flowers, tested on {len(yte)} unseen flowers")
print(f"Train accuracy {knn.score(Xtr, ytr):.3f} | Test accuracy {knn.score(Xte, yte):.3f}"
      "  <- the test score measures GENERALISATION")

# --- 3. Regression: predict a NUMBER (salary in $1000s) from years of experience
years = np.array([1, 3, 4, 6, 7, 9, 11, 13, 16, 20]).reshape(-1, 1)
salary = np.array([35, 40, 47, 50, 56, 58, 67, 71, 82, 93])
reg = LinearRegression().fit(years, salary)
print(f"\nSalary line: salary = {reg.coef_[0]:.2f} x years + {reg.intercept_:.2f}")
print(f"Predicted salary for 10 years: {reg.predict([[10]])[0]:.2f} thousand")
