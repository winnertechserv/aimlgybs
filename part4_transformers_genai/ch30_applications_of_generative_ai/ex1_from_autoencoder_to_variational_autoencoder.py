"""
Chapter 30, Example 1: From Autoencoder to Variational Autoencoder
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This contrasts the deterministic AANN, which maps an input to a single bottleneck point,
with the VAE, whose encoder outputs a mean and log-variance. It then shows the
reparameterization trick, z = mu + sigma*eps, showing that the same input yields
different latent samples each time, and computes both parts of the VAE loss including
the KL term that pulls the latent distribution towards N(0,I).

Run it:           python ex1_from_autoencoder_to_variational_autoencoder.py
Expected output:  expected_output/ex1_from_autoencoder_to_variational_autoencoder.txt
"""
# Chapter 30 - Example 1: Autoencoder -> VAE (reparameterization trick + KL)
import numpy as np
np.random.seed(0)
np.set_printoptions(suppress=True)

print("=== From Autoencoder to Variational Autoencoder ===\n")
print("AANN / deterministic autoencoder: encoder maps x to a SINGLE point")
print("in the bottleneck (dimension-reduction) layer.")
print("VAE: encoder maps x to a DISTRIBUTION (mean + std) in latent space,")
print("and the decoder samples from it -> we can GENERATE new data.\n")

d, latent = 8, 3
x = np.round(np.random.randn(d), 2)
print(f"Input image x ({d}-dim): {x}\n")

# ---- Deterministic autoencoder ----
W_enc = np.random.randn(d, latent) * 0.4
W_dec = np.random.randn(latent, d) * 0.4
h = np.tanh(x @ W_enc)
x_rec = h @ W_dec
print("--- Deterministic Autoencoder (AANN) ---")
print(f"  bottleneck code h : {np.round(h, 4)}   (one fixed point)")
print(f"  reconstruction    : {np.round(x_rec, 3)}")
print(f"  reconstruction MSE: {np.mean((x - x_rec)**2):.4f}")
print("  Sampling a random h gives arbitrary output - the latent space")
print("  has no structure, so this model is a poor generator.\n")

# ---- VAE encoder: mean and log-variance ----
print("--- Variational Autoencoder (VAE) ---\n")
W_mu     = np.random.randn(d, latent) * 0.4
W_logvar = np.random.randn(d, latent) * 0.4
mu     = x @ W_mu
logvar = x @ W_logvar
sigma  = np.exp(0.5 * logvar)
print("Step 1 - encoder outputs a distribution per dimension:")
print(f"  mu     = {np.round(mu, 4)}")
print(f"  logvar = {np.round(logvar, 4)}")
print(f"  sigma  = exp(0.5*logvar) = {np.round(sigma, 4)}\n")

# ---- Reparameterization trick ----
print("Step 2 - REPARAMETERIZATION TRICK:  z = mu + sigma * epsilon,  eps~N(0,I)")
print("  (sampling directly from N(mu,sigma) is not differentiable;")
print("   moving the randomness into eps makes backpropagation possible)\n")
for trial in range(3):
    eps = np.random.randn(latent)
    z = mu + sigma * eps
    print(f"  draw {trial+1}: eps = {np.round(eps,3)} -> z = {np.round(z,4)}")
print("  Same input x, different z each time -> the decoder generates variety.\n")

# ---- VAE loss ----
print("Step 3 - VAE loss = reconstruction loss + KL divergence\n")
eps = np.random.randn(latent)
z = mu + sigma * eps
x_hat = z @ W_dec
recon = np.mean((x - x_hat) ** 2)
kl = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
beta = 1.0
print(f"  Reconstruction loss (MSE)          : {recon:.4f}")
print(f"  KL( q(z|x) || N(0,I) )             : {kl:.4f}")
print(f"  Total loss = recon + beta*KL       : {recon + beta*kl:.4f}")
print("\n  The KL term pulls the latent distribution towards N(0,I), which")
print("  makes the latent space smooth and continuous - so sampling")
print("  z ~ N(0,I) at test time produces realistic new images.\n")

print("KL for special cases (per dimension):")
for m, lv in [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, -2.0)]:
    k = -0.5 * (1 + lv - m**2 - np.exp(lv))
    note = ("perfect match to N(0,1)" if (m, lv) == (0.0, 0.0)
            else "mean far from 0" if m else "variance mismatch")
    print(f"  mu={m:+.1f}, logvar={lv:+.1f} -> KL = {k:.4f}   ({note})")

print("\n--- Denoising Autoencoder ---")
print("  Trained on (noisy image, clean image) pairs; at test time a noisy")
print("  input produces a denoised output. The decoder learns to GENERATE.")
