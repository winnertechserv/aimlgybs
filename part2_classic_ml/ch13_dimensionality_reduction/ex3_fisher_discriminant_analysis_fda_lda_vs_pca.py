"""
Chapter 13, Example 3: Fisher Discriminant Analysis (FDA/LDA) vs PCA
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This is the key example that proves WHY FDA beats PCA for classification tasks. We use
sklearn's LinearDiscriminantAnalysis (which builds FDA) and compare it to PCA on the
same 10-feature, 2-class dataset. Both reduce to 1 dimension. FDA finds a direction that
maximises class separation (the class means are 1.96 apart after projection). PCA finds
the direction of maximum variance — but that direction might not be useful for
distinguishing classes. The accuracy comparison confirms it: FDA 1D is nearly as good as
the full 10D dataset, while PCA 1D loses large accuracy.

Run it:           python ex3_fisher_discriminant_analysis_fda_lda_vs_pca.py
Expected output:  expected_output/ex3_fisher_discriminant_analysis_fda_lda_vs_pca.txt
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
X, y = make_classification(n_samples=300, n_features=10, n_informative=5,
                            n_redundant=3, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("=== FDA vs PCA for Classification ===\n")
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, 2 classes\n")

# FDA: project to 1D (max class separability)
lda = LinearDiscriminantAnalysis(n_components=1)
X_tr_lda = lda.fit_transform(X_train, y_train)
X_te_lda = lda.transform(X_test)

print("Step 1: FDA projection direction maximises class separation")
print(f"  Class +1 mean in FDA space: {X_tr_lda[y_train==1].mean():.4f}")
print(f"  Class  0 mean in FDA space: {X_tr_lda[y_train==0].mean():.4f}")
print(f"  Separation between means  : {abs(X_tr_lda[y_train==1].mean() - X_tr_lda[y_train==0].mean()):.4f}")

# PCA: project to 1D (max variance)
pca = PCA(n_components=1)
X_tr_pca = pca.fit_transform(X_train)
X_te_pca = pca.transform(X_test)

def knn_acc(X_tr, X_te, y_tr, y_te):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te))

print("\nStep 2: Classification accuracy comparison (5-NN classifier)")
print(f"  Original 10D data      : {knn_acc(X_train, X_test, y_train, y_test)*100:.1f}%")
print(f"  PCA 1D projection      : {knn_acc(X_tr_pca, X_te_pca, y_train, y_test)*100:.1f}%  (max variance, may miss class info)")
print(f"  FDA 1D projection      : {knn_acc(X_tr_lda, X_te_lda, y_train, y_test)*100:.1f}%  (max separability)")

print("\nFDA projection weights per feature:")
for i, w in enumerate(lda.coef_[0]):
    print(f"  Feature {i+1:>2}: {w:+.4f}")
