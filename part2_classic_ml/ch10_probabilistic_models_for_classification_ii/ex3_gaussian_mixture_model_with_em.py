"""
Chapter 10, Example 3: Gaussian Mixture Model with EM
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Class A has two separate clusters. We train a 2-component GMM with EM written out in
full. Watch the log-likelihood rise and then stop changing after four iterations, and
the learned centers land on the two true clusters at (0, 0) and (4, 4). In the Bayes
classifier, a single Gaussian for class A scores 95% while a GMM scores 99%. The final
table is the elbow: the log-likelihood jumps from K = 1 to K = 2, then barely improves,
so K = 2 is the right choice.

Run it:           python ex3_gaussian_mixture_model_with_em.py
Expected output:  expected_output/ex3_gaussian_mixture_model_with_em.txt
"""
# Chapter 10 - Example 3: Gaussian Mixture Model trained with EM, used in a Bayes classifier
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

rng = np.random.default_rng(0)
# Class A has TWO clusters (multimodal); class B sits in between
A = np.vstack([rng.normal([0, 0], 0.6, (150, 2)), rng.normal([4, 4], 0.6, (150, 2))])
B = rng.normal([2, 2], 0.6, (300, 2))

# --- EM from scratch on class A with K = 2 components
K = 2
mu = A[rng.choice(len(A), K, replace=False)]          # random initial centers
cov = np.array([np.eye(2)] * K)
w = np.full(K, 1 / K)
prev = -np.inf
for it in range(1, 50):
    dens = np.column_stack([w[k] * multivariate_normal(mu[k], cov[k]).pdf(A) for k in range(K)])
    ll = np.log(dens.sum(axis=1)).sum()                # total data log-likelihood
    gamma = dens / dens.sum(axis=1, keepdims=True)     # E-step: responsibilities
    Nk = gamma.sum(axis=0)                             # M-step: re-estimate parameters
    mu = (gamma.T @ A) / Nk[:, None]
    cov = np.array([((gamma[:, k, None] * (A - mu[k])).T @ (A - mu[k])) / Nk[k]
                    for k in range(K)])
    w = Nk / len(A)
    if it <= 4 or abs(ll - prev) < 1e-3:
        print(f"EM iteration {it:2d}: log-likelihood = {ll:9.2f}")
    if abs(ll - prev) < 1e-3:
        print("Converged (change < 0.001).")
        break
    prev = ll
print("Learned centers:", np.round(mu, 2).tolist(), " weights:", np.round(w, 2).tolist())

# --- Bayes classifier: one density per class, single Gaussian vs GMM
X_te = np.vstack([rng.normal([0, 0], 0.6, (100, 2)), rng.normal([4, 4], 0.6, (100, 2)),
                  rng.normal([2, 2], 0.6, (200, 2))])
y_te = np.r_[np.zeros(200), np.ones(200)]
for k in (1, 2):
    gA = GaussianMixture(k, random_state=0).fit(A)
    gB = GaussianMixture(1, random_state=0).fit(B)
    pred = (gB.score_samples(X_te) > gA.score_samples(X_te)).astype(int)  # equal priors
    print(f"Class A modeled with K = {k} component(s): test accuracy {np.mean(pred == y_te):.3f}")

# --- Elbow: total log-likelihood of class A for different K
print("\nK   total log-likelihood")
for k in range(1, 6):
    g = GaussianMixture(k, random_state=0).fit(A)
    print(f"{k}   {g.score(A) * len(A):9.1f}")
