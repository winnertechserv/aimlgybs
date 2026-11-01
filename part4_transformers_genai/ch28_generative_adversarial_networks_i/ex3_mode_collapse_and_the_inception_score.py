"""
Chapter 28, Example 3: Mode Collapse and the Inception Score
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
The first half shows mode collapse: a healthy GAN covers all ten digit modes while a
collapsed one produces only two. The second half builds the Inception Score using KL
divergence and shows that a high score requires BOTH confident per-image predictions AND
a diverse marginal distribution - mode collapse destroys the second, blurry generation
destroys the first.

Run it:           python ex3_mode_collapse_and_the_inception_score.py
Expected output:  expected_output/ex3_mode_collapse_and_the_inception_score.txt
"""
# Chapter 28 - Example 3: Mode collapse + Inception Score (GAN evaluation)
import numpy as np
np.random.seed(5)

print("=== Evaluating a GAN: Creativity, Inheritance, Diversity ===\n")
print("A GAN image generator is judged on three aspects:")
print("  Creativity  : generated images are NOT duplicates of real images")
print("  Inheritance : generated images retain key features of real images")
print("  Diversity   : generated images differ from one another\n")

# ---------- Mode collapse ----------
print("=== Mode Collapse (Mode Diversity Problem) ===\n")
print("The generator maps ALL noise vectors z to a small subset of outputs.")
print("Example: a GAN trained on handwritten digits generates only a few digits.\n")

n = 1000
healthy  = np.random.choice(10, n)                       # all 10 digits
collapsed = np.random.choice([3, 8], n, p=[0.6, 0.4])    # only 2 digits

def digit_hist(labels, title):
    counts = np.bincount(labels, minlength=10)
    print(f"  {title}")
    for d in range(10):
        bar = "#" * int(counts[d] / 8)
        print(f"    digit {d}: {counts[d]:>4}  {bar}")
    modes = int((counts > 0).sum())
    print(f"    distinct modes generated: {modes}/10\n")
    return counts

c_ok  = digit_hist(healthy,   "Healthy GAN (good mode coverage):")
c_bad = digit_hist(collapsed, "Mode-collapsed GAN:")

print("The fix: the Least-Squares GAN (LSGAN) replaces the")
print("cross-entropy discriminator loss with sum-of-squared errors and is")
print("shown to avoid mode collapse. The Wasserstein GAN instead changes")
print("the criterion to the Wasserstein (Earth Mover's) distance, which")
print("measures how far apart p(x~) and p(x) are.\n")

# ---------- Inception Score ----------
print("=== Inception Score (IS) via KL Divergence ===\n")
print("Generated images are fed to an Inception network. For each image we get")
print("p(y|x) over class labels. IS = exp( E_x [ KL( p(y|x) || p(y) ) ] )\n")

def inception_score(P):
    """P: (n_images x n_classes) matrix of class probabilities."""
    p_y = P.mean(axis=0)                                  # marginal p(y)
    kl  = np.sum(P * (np.log(P + 1e-12) - np.log(p_y + 1e-12)), axis=1)
    return float(np.exp(kl.mean())), p_y

K = 5
# Case 1: sharp per-image predictions AND diverse classes -> high IS
sharp_diverse = np.full((100, K), 0.02)
for i in range(100):
    sharp_diverse[i, i % K] = 0.92
# Case 2: sharp but ALL the same class (mode collapse) -> low IS
sharp_collapsed = np.full((100, K), 0.02)
sharp_collapsed[:, 0] = 0.92
# Case 3: blurry/uncertain predictions -> low IS
blurry = np.full((100, K), 1.0 / K)

for name, P in [("Sharp + diverse (good GAN)", sharp_diverse),
                ("Sharp but mode-collapsed",   sharp_collapsed),
                ("Blurry / unrecognisable",    blurry)]:
    P = P / P.sum(axis=1, keepdims=True)
    score, p_y = inception_score(P)
    print(f"  {name:<30} IS = {score:.4f}   marginal p(y) = {np.round(p_y,3)}")

print(f"\n  Maximum possible IS for {K} classes = {K}")
print("\nInterpretation:")
print("  High IS needs BOTH  (a) confident per-image class predictions")
print("                 and  (b) a uniform marginal over classes (diversity).")
print("  Mode collapse kills (b), blurry generation kills (a).")
print("\nThe other standard metric is the Frechet Inception Distance (FID),")
print("which compares the feature distributions of real and generated images.")
