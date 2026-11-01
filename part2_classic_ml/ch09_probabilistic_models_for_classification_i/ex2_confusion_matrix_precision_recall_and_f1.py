"""
Chapter 9, Example 2: Confusion Matrix, Precision, Recall and F1
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We classify breast tumors as malignant (positive) or benign, using K-NN on a stratified
70/30 split. The program prints the confusion matrix, then works out accuracy,
precision, recall and F1 by hand and checks them against scikit-learn. Both agree. Look
at the false negatives: 4 cancers were missed. That number is why recall is the metric
doctors care about most.

Run it:           python ex2_confusion_matrix_precision_recall_and_f1.py
Expected output:  expected_output/ex2_confusion_matrix_precision_recall_and_f1.txt
"""
# Chapter 9 - Example 2: Evaluating a classifier - confusion matrix, accuracy,
# precision, recall and F1, with a stratified 70/30 split
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

data = load_breast_cancer()
X, y = data.data, 1 - data.target        # make 1 = malignant (the 'positive' class we must catch)
print(f"Samples: {len(y)}  (malignant {y.sum()}, benign {len(y) - y.sum()})")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print(f"Train {len(y_tr)} / Test {len(y_te)}; malignant share train "
      f"{y_tr.mean():.2f}, test {y_te.mean():.2f} (stratified = same mix)")

sc = StandardScaler().fit(X_tr)
model = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(X_tr), y_tr)
pred = model.predict(sc.transform(X_te))

tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print("\nConfusion matrix          Actual +   Actual -")
print(f"  Predicted + (malignant)   TP={tp:3d}     FP={fp:3d}")
print(f"  Predicted - (benign)      FN={fn:3d}     TN={tn:3d}")

acc = (tp + tn) / (tp + tn + fp + fn)
prec = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = 2 * prec * rec / (prec + rec)
print(f"\nBy hand : accuracy {acc:.3f}  precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}")
print(f"sklearn : precision {precision_score(y_te, pred):.3f}  recall "
      f"{recall_score(y_te, pred):.3f}  F1 {f1_score(y_te, pred):.3f}")
print(f"\n{fn} cancer case(s) were missed (false negatives). In medicine, recall matters most.")
