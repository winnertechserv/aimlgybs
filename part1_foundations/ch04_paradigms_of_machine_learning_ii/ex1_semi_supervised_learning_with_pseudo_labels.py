"""
Chapter 4, Example 1: Semi-Supervised Learning with Pseudo-Labels
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Only 50 of 1,257 digit images keep their labels. A model trained on those 50 scores
82.6%. Then the self-training loop runs. Each round, the model labels the unlabeled
images it is at least 90% sure about and retrains on them. Within four rounds it has
pseudo-labeled over 1,000 images and test accuracy rises to 88.1%. Label spreading, a
second semi-supervised method, passes labels to neighboring images and reaches 92.8%
from the same 50 labels. Compare these with the 96.3% you get when all 1,257 labels are
available.

Run it:           python ex1_semi_supervised_learning_with_pseudo_labels.py
Expected output:  expected_output/ex1_semi_supervised_learning_with_pseudo_labels.txt
"""
# Chapter 4 - Example 1: Semi-supervised learning with pseudo-labels (self-training)
# Only 50 of ~1,250 training images have labels. The model labels the rest itself,
# keeps only the labels it is very confident about, and retrains.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading

X, y = load_digits(return_X_y=True)
X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

rng = np.random.default_rng(0)
labeled = np.zeros(len(X_tr), bool)
for d in range(10):                                  # 5 labeled images per digit
    labeled[rng.choice(np.where(y_tr == d)[0], 5, replace=False)] = True
X_l, y_l = X_tr[labeled], y_tr[labeled]
X_u = X_tr[~labeled]                                # unlabeled pool (we ignore y for these)
print(f"Labeled: {len(X_l)}   Unlabeled: {len(X_u)}")

def new_model():
    return LogisticRegression(C=100, max_iter=3000)

model = new_model().fit(X_l, y_l)
print(f"Round 0 (labeled data only): test accuracy = {model.score(X_te, y_te):.3f}")

for rnd in range(1, 5):
    proba = model.predict_proba(X_u)
    confident = proba.max(axis=1) >= 0.9             # keep only very confident guesses
    pseudo = model.classes_[proba.argmax(axis=1)]
    model = new_model().fit(np.vstack([X_l, X_u[confident]]),
                            np.r_[y_l, pseudo[confident]])
    true_u = y_tr[~labeled][confident]               # peek ONLY to report pseudo-label quality
    print(f"Round {rnd}: pseudo-labeled {confident.sum():4d} images "
          f"(pseudo-label accuracy {np.mean(pseudo[confident] == true_u):.3f}) "
          f"-> test accuracy = {model.score(X_te, y_te):.3f}")

# A second semi-supervised method: label spreading passes labels along to near neighbors
y_partial = np.where(labeled, y_tr, -1)             # -1 means 'no label'
ls = LabelSpreading(kernel="knn", n_neighbors=10).fit(X_tr, y_partial)
print(f"\nLabel spreading (same 50 labels): test accuracy = {ls.score(X_te, y_te):.3f}")

full = new_model().fit(X_tr, y_tr)
print(f"For reference, with ALL {len(X_tr)} true labels: {full.score(X_te, y_te):.3f}")
