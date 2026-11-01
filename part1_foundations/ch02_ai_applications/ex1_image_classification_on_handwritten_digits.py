"""
Chapter 2, Example 1: Image Classification on Handwritten Digits
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Scikit-learn includes 1,797 tiny 8x8 images of handwritten digits. Each image is 64
pixel values, so to the computer an image is just a list of 64 numbers. The program
draws one digit as text so you can see it, trains a support vector machine (you will
meet SVMs in Chapters 11 and 12) on 70% of the images, and tests on the rest. It reaches
99.4% accuracy and reports which digits it confused.

Run it:           python ex1_image_classification_on_handwritten_digits.py
Expected output:  expected_output/ex1_image_classification_on_handwritten_digits.txt
"""
# Chapter 2 - Example 1: Image classification on handwritten digits
# Each 8x8 image becomes 64 numbers (pixel intensities); a classifier learns the digits.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

digits = load_digits()                      # 1,797 images, built into scikit-learn
X, y = digits.data, digits.target
print("Images:", X.shape[0], "| pixels per image:", X.shape[1], "| classes:", sorted(set(y.tolist())))

# Show one image as text so we can 'see' it
img = digits.images[0]
print(f"\nImage 0 (label {y[0]}):")
for row in img:
    print("  " + "".join("#" if v > 8 else "+" if v > 3 else "." for v in row))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
clf = SVC(kernel="rbf", gamma=0.001, C=10).fit(X_tr, y_tr)
pred = clf.predict(X_te)
print(f"\nTest accuracy: {accuracy_score(y_te, pred):.3f} on {len(y_te)} unseen images")

cm = confusion_matrix(y_te, pred)
off = cm.copy(); np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"Most common mistake: true {i} predicted as {j} ({off[i, j]} time(s))")
print("Per-digit accuracy:", {d: round(float(cm[d, d] / cm[d].sum()), 2) for d in range(10)})
