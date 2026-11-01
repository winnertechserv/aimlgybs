"""
Chapter 28, Example 1: The GAN Minimax Objective
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This computes the GAN objective V(D,G) numerically for several scenarios, showing it is
high when the discriminator wins and low when the generator wins, with the theoretical
equilibrium at 2*log(0.5) = -1.386. The second half walks through one mini-batch
training step exactly as the algorithm slide describes, verifying that the
discriminator's gradient ascent raises V and the generator's gradient descent lowers it.

Run it:           python ex1_the_gan_minimax_objective.py
Expected output:  expected_output/ex1_the_gan_minimax_objective.txt
"""
# Chapter 28 - Example 1: GAN minimax objective computed numerically
import numpy as np
np.random.seed(0)

print("=== The GAN Minimax Optimization Problem ===\n")
print("min_thetaG  max_thetaD   E_p(x)[log D(x)] + E_p(z)[log(1 - D(G(z)))]\n")
print("  Discriminator D: MAXIMIZE it (gradient ASCENT)")
print("  Generator     G: MINIMIZE it (gradient DESCENT)")
print("  Desired D output: 1 for a real image, 0 for a synthetic image\n")

def objective(D_real, D_fake):
    """V(D,G) for a mini-batch"""
    return np.mean(np.log(D_real + 1e-12)) + np.mean(np.log(1 - D_fake + 1e-12))

M = 6   # mini-batch size
scenarios = {
  "Untrained D (outputs ~0.5 for everything)":
      (np.full(M, 0.50), np.full(M, 0.50)),
  "Strong D (real->0.95, fake->0.05): G is losing":
      (np.full(M, 0.95), np.full(M, 0.05)),
  "Strong G (D fooled: fake->0.90)":
      (np.full(M, 0.60), np.full(M, 0.90)),
  "Equilibrium (D cannot tell: both 0.5)":
      (np.full(M, 0.50), np.full(M, 0.50)),
}

print(f"{'Scenario':<48} {'V(D,G)':>10}  {'Reading'}")
print("-" * 88)
for name, (dr, df) in scenarios.items():
    V = objective(dr, df)
    if "Strong D" in name:   read = "D winning (V high)"
    elif "Strong G" in name: read = "G winning (V low)"
    else:                    read = "V = 2*log(0.5) = -1.386"
    print(f"  {name:<46} {V:>10.4f}  {read}")

print(f"\nTheoretical equilibrium value: 2*log(0.5) = {2*np.log(0.5):.4f}")
print("At equilibrium p(x_synthetic) ~ p(x_real), so D outputs 0.5 everywhere.\n")

# ---- Mini-batch training step (the standard algorithm) ----
print("=== Mini-Batch Mode Training of a GAN (one step) ===\n")
theta_D, theta_G = 0.0, 0.0     # scalar stand-ins for parameter vectors
eta = 0.1
D_real = np.random.uniform(0.55, 0.75, M)
D_fake = np.random.uniform(0.25, 0.45, M)

print("Step 3.1  Sample M random noise vectors z ~ N(0, sigma^2 I)")
z = np.random.randn(M, 4)
print(f"          z shape = {z.shape}  (M={M} noise vectors of dim 4)")

print("\nStep 3.2  Update DISCRIMINATOR by GRADIENT ASCENT")
V_before = objective(D_real, D_fake)
# ascent: D pushes D(x)->1 and D(x~)->0
D_real_new = np.clip(D_real + eta * (1 - D_real), 0, 1)
D_fake_new = np.clip(D_fake - eta * D_fake, 0, 1)
V_after_D = objective(D_real_new, D_fake_new)
print(f"          V before = {V_before:.4f}")
print(f"          V after  = {V_after_D:.4f}   (INCREASED -> ascent worked)")

print("\nStep 3.3  Update GENERATOR by GRADIENT DESCENT")
# descent: G pushes D(x~)->1
D_fake_G = np.clip(D_fake_new + 3 * eta * (1 - D_fake_new), 0, 1)
V_after_G = objective(D_real_new, D_fake_G)
print(f"          V before = {V_after_D:.4f}")
print(f"          V after  = {V_after_G:.4f}   (DECREASED -> descent worked)")

print("\nStep 4    Repeat until convergence.")
print("\nNote the adversarial tension: D pushes V up, G pushes V down.")
print("This is why the objective goes up AND down during GAN training,")
print("unlike ordinary supervised learning where loss decreases steadily.")
