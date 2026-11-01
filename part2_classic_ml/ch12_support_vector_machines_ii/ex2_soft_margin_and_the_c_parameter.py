"""
Chapter 12, Example 2: Soft Margin and the C Parameter
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Two overlapping classes, so a soft margin is needed. As C grows from 0.001 to 100, the
margin shrinks from 4.03 to 0.55 and the number of support vectors falls from 178 to 36.
The model trusts fewer points and fits the training set harder. Test accuracy stays
about the same, which shows that a bigger C is not automatically better. The last lines
sort the training points by hinge loss: outside the margin, inside it, or on the wrong
side.

Run it:           python ex2_soft_margin_and_the_c_parameter.py
Expected output:  expected_output/ex2_soft_margin_and_the_c_parameter.txt
"""
# Chapter 12 - Example 2: Soft margin - the C hyperparameter and the hinge loss
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([0, 0], 1.0, (150, 2)), rng.normal([2, 2], 1.0, (150, 2))])
y = np.r_[-np.ones(150), np.ones(150)]              # classes overlap a little
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)

print("    C     margin  support vectors  train acc  test acc  points with xi>1")
for C in (0.001, 0.01, 0.1, 1, 100):
    m = SVC(kernel="linear", C=C).fit(X_tr, y_tr)
    w, w0 = m.coef_[0], m.intercept_[0]
    xi = np.maximum(0, 1 - y_tr * (X_tr @ w + w0))  # hinge loss of each training point
    print(f"  {C:<7g}  {1 / np.linalg.norm(w):5.2f}       {len(m.support_):3d}          "
          f"{m.score(X_tr, y_tr):.3f}     {m.score(X_te, y_te):.3f}       {np.sum(xi > 1):3d}")
print("\nSmall C: wide margin, many support vectors, tolerates errors.")
print("Large C: narrow margin, tries hard to classify every training point.")

# What the hinge loss means for three individual points (C = 1 model)
m = SVC(kernel="linear", C=1).fit(X_tr, y_tr)
g = X_tr @ m.coef_[0] + m.intercept_[0]
xi = np.maximum(0, 1 - y_tr * g)
for label, mask in (("outside the margin (xi = 0)", xi == 0),
                    ("inside the margin (0 < xi < 1)", (xi > 0) & (xi < 1)),
                    ("wrong side (xi > 1)", xi > 1)):
    print(f"  {label:32s}: {mask.sum():3d} points")
