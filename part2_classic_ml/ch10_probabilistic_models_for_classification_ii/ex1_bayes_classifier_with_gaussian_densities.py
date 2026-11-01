"""
Chapter 10, Example 1: Bayes Classifier with Gaussian Densities
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We fit one Gaussian per class (mean and covariance) to the adult/child data, then apply
Bayes' rule step by step: likelihood, then evidence, then posterior. The likelihoods for
the test person are 3.44×10⁻¹⁷ (child) and 5.31×10⁻⁴ (adult), so adult wins
overwhelmingly. The second part shows why priors matter. The borderline person (140 cm,
35 kg) is 'Adult' with equal priors but 'Child' once we use the priors of a primary
school.

Run it:           python ex1_bayes_classifier_with_gaussian_densities.py
Expected output:  expected_output/ex1_bayes_classifier_with_gaussian_densities.txt
"""
# Chapter 10 - Example 1: Bayes classifier with a Gaussian density for each class
import numpy as np

child = np.array([[98, 16.4], [104, 15.9], [110, 19.8], [115, 19.1], [119, 23.0],
                  [123, 22.4], [127, 27.1], [131, 26.3], [134, 31.0], [101, 17.5]])
adult = np.array([[152, 52.0], [158, 50.5], [161, 63.2], [165, 57.8], [168, 71.0],
                  [172, 64.5], [175, 80.3], [179, 72.9], [183, 88.0], [188, 81.4]])

def gaussian_pdf(x, mu, cov):
    d = len(mu)
    diff = x - mu
    norm = 1.0 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(cov)))
    return norm * np.exp(-0.5 * diff @ np.linalg.inv(cov) @ diff)

# Training: prior + Gaussian parameters (mean, covariance) for each class
N = len(child) + len(adult)
classes = {}
for name, data in (("Child", child), ("Adult", adult)):
    classes[name] = dict(prior=len(data) / N, mu=data.mean(axis=0),
                         cov=np.cov(data, rowvar=False))
    print(f"{name}: prior P = {classes[name]['prior']:.2f}, mean = "
          f"{np.round(classes[name]['mu'], 2)}")

def classify(x, priors=None):
    like = {c: gaussian_pdf(x, p["mu"], p["cov"]) for c, p in classes.items()}
    pri = priors or {c: p["prior"] for c, p in classes.items()}
    evidence = sum(like[c] * pri[c] for c in classes)          # total probability P(x)
    post = {c: like[c] * pri[c] / evidence for c in classes}   # Bayes rule
    return like, post

x = np.array([150, 47.0])
like, post = classify(x)
print(f"\nTest person x = {x}")
for c in classes:
    print(f"  likelihood p(x | {c}) = {like[c]:.2e}   posterior P({c} | x) = {post[c]:.2e}")
print("  Decision:", max(post, key=post.get))

# The prior matters: data collected in a primary school (9 children for every adult)
x2 = np.array([140, 35.0])
for label, pri in (("equal priors", None), ("school priors", {"Child": 0.9, "Adult": 0.1})):
    _, post = classify(x2, pri)
    print(f"\nx2 = {x2} with {label}: P(Child|x) = {post['Child']:.3f}, "
          f"P(Adult|x) = {post['Adult']:.3f} -> {max(post, key=post.get)}")
