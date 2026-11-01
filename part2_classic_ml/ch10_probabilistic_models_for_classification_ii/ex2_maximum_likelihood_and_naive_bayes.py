"""
Chapter 10, Example 2: Maximum Likelihood and Naive Bayes
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
First we run a grid search: try thousands of candidate means for the child Gaussian and
keep the one with the highest log-likelihood. It lands on the sample mean, which shows
that the ML estimate is the sample mean. Then we show why logs are used: the raw
likelihood of just 10 points is about 10⁻²⁵. Finally, on the Wine dataset (13 features,
3 classes), naive Bayes uses 26 parameters per class against 104 for a full covariance.
It is slightly less accurate here because it ignores correlations.

Run it:           python ex2_maximum_likelihood_and_naive_bayes.py
Expected output:  expected_output/ex2_maximum_likelihood_and_naive_bayes.txt
"""
# Chapter 10 - Example 2: Maximum likelihood estimation, and Naive Bayes vs full Bayes
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

child = np.array([[98, 16.4], [104, 15.9], [110, 19.8], [115, 19.1], [119, 23.0],
                  [123, 22.4], [127, 27.1], [131, 26.3], [134, 31.0], [101, 17.5]])
cov = np.cov(child, rowvar=False)

# 1. Search a grid of possible means; keep the one with the highest log-likelihood
best, best_ll = None, -np.inf
for m1 in np.linspace(66.2, 166.2, 201):
    for m2 in np.linspace(-28.15, 71.85, 201):
        ll = multivariate_normal([m1, m2], cov).logpdf(child).sum()  # sum of logs
        if ll > best_ll:
            best, best_ll = (m1, m2), ll
print(f"Grid search best mean : ({best[0]:.2f}, {best[1]:.2f})  log-likelihood {best_ll:.2f}")
print(f"Sample mean (formula) : ({child[:, 0].mean():.2f}, {child[:, 1].mean():.2f})")
print("-> Maximum likelihood estimate of a Gaussian mean = the sample mean.\n")

# 2. Why use logs? The raw likelihood is a product of tiny numbers
p = multivariate_normal(child.mean(0), cov).pdf(child)
print(f"Product of 10 likelihoods: {np.prod(p):.3e}  (sum of logs: {np.log(p).sum():.2f})\n")

# 3. Naive Bayes (diagonal covariance) vs full-covariance Bayes on the Wine dataset
X, y = load_wine(return_X_y=True)              # 178 wines, 13 chemical features, 3 classes
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
d = X.shape[1]
nb = GaussianNB().fit(Xtr, ytr)
full = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(Xtr, ytr)
print(f"Parameters per class: Naive Bayes {2 * d}, full covariance {d + d * (d + 1) // 2}")
print(f"Test accuracy  Naive Bayes: {nb.score(Xte, yte):.3f}   "
      f"Full-covariance Bayes: {full.score(Xte, yte):.3f}")
