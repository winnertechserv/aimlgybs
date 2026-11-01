"""
Chapter 9, Example 3: Reference Templates — Euclidean vs Mahalanobis
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Training is just computing each class's mean and covariance: the child mean is [116.2,
21.85] and the adult mean is [170.1, 68.16]. For the first test person both distances
agree: Euclidean 42.13 vs 29.18, Mahalanobis 8.13 vs 1.76, so the answer is adult. The
second test person (137 cm, 38 kg) shows the two distances disagreeing. In plain
distance they are nearer the children (26.33 vs 44.78). But children's weight follows
their height very closely, and 38 kg is unusually heavy for a 137 cm child. For adults,
38 kg at 137 cm sits right on the usual height-weight trend. Mahalanobis takes that
pattern into account and picks adult (2.94 vs 5.40).

Run it:           python ex3_reference_templates_euclidean_vs_mahalanobis.py
Expected output:  expected_output/ex3_reference_templates_euclidean_vs_mahalanobis.txt
"""
# Chapter 9 - Example 3: Reference templates - Euclidean vs Mahalanobis distance
import numpy as np
np.set_printoptions(suppress=True)

child = np.array([[98, 16.4], [104, 15.9], [110, 19.8], [115, 19.1], [119, 23.0],
                  [123, 22.4], [127, 27.1], [131, 26.3], [134, 31.0], [101, 17.5]])
adult = np.array([[152, 52.0], [158, 50.5], [161, 63.2], [165, 57.8], [168, 71.0],
                  [172, 64.5], [175, 80.3], [179, 72.9], [183, 88.0], [188, 81.4]])

# Training = computing statistics of each class
templates = {}
for name, data in (("Child", child), ("Adult", adult)):
    mu = data.mean(axis=0)                    # first-order statistic: the mean
    cov = np.cov(data, rowvar=False)          # second-order: covariance (divides by N-1)
    templates[name] = (mu, cov)
    print(f"{name}: mean = {np.round(mu, 2)}, covariance =\n{np.round(cov, 2)}\n")

x = np.array([150, 47.0])
print(f"Test person x = {x}")
for name, (mu, cov) in templates.items():
    ed = np.sqrt(((x - mu) ** 2).sum())
    md = np.sqrt((x - mu) @ np.linalg.inv(cov) @ (x - mu))
    print(f"  {name:5s}: Euclidean distance {ed:6.2f}   Mahalanobis distance {md:5.2f}")

# A case where the two distances disagree
x2 = np.array([137, 38.0])
print(f"\nTest person x2 = {x2} (between the two groups)")
res = {}
for name, (mu, cov) in templates.items():
    ed = np.sqrt(((x2 - mu) ** 2).sum())
    md = np.sqrt((x2 - mu) @ np.linalg.inv(cov) @ (x2 - mu))
    res[name] = (ed, md)
    print(f"  {name:5s}: Euclidean {ed:6.2f}   Mahalanobis {md:5.2f}")
print("  Euclidean picks:", min(res, key=lambda k: res[k][0]),
      "| Mahalanobis picks:", min(res, key=lambda k: res[k][1]))
