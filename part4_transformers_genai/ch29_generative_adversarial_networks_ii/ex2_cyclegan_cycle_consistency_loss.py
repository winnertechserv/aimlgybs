"""
Chapter 29, Example 2: CycleGAN Cycle-Consistency Loss
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This shows the idea that makes unpaired translation possible. An image is translated to
the other domain and back; a well-trained pair of generators returns the original almost
exactly (loss near 0), while poorly trained ones do not. The full objective combines
both adversarial losses with the weighted cycle loss, and the SuperstarGAN FID table
from the slide is reproduced at the end.

Run it:           python ex2_cyclegan_cycle_consistency_loss.py
Expected output:  expected_output/ex2_cyclegan_cycle_consistency_loss.txt
"""
# Chapter 29 - Example 2: CycleGAN cycle-consistency loss
import numpy as np
np.random.seed(1)
np.set_printoptions(suppress=True)

print("=== CycleGAN: Unpaired Image-to-Image Translation ===\n")
print("pix2pix needs PAIRED data (same scene in both domains).")
print("CycleGAN works with UNPAIRED data using two generators and two")
print("discriminators, tied together by the cycle-consistency loss.\n")
print("  G_AB : domain A -> domain B      D_B : judges domain B images")
print("  G_BA : domain B -> domain A      D_A : judges domain A images\n")
print("Cycle consistency:  G_BA(G_AB(a)) ~ a    and    G_AB(G_BA(b)) ~ b\n")

# Represent tiny 3x3 'images'
a = np.round(np.random.rand(3, 3), 2)          # image in domain A (e.g. horse)
print("Original image a (domain A):"); print(a)

# Simulate two generators as invertible-ish transforms
def G_AB(x, quality=1.0):
    """A -> B (e.g. horse -> zebra). quality=1 means perfectly invertible."""
    noise = (1 - quality) * np.random.randn(*x.shape) * 0.3
    return np.clip(x * 0.8 + 0.15 + noise, 0, 1)

def G_BA(y, quality=1.0):
    """B -> A (inverse mapping)."""
    noise = (1 - quality) * np.random.randn(*y.shape) * 0.3
    return np.clip((y - 0.15) / 0.8 + noise, 0, 1)

print("\n--- Well-trained CycleGAN (quality = 1.0) ---")
b_hat = G_AB(a, 1.0)
a_cyc = G_BA(b_hat, 1.0)
print("Translated to domain B, G_AB(a):"); print(np.round(b_hat, 3))
print("Cycled back, G_BA(G_AB(a)):");      print(np.round(a_cyc, 3))
L_cyc_good = np.abs(a - a_cyc).mean()
print(f"Cycle-consistency loss  L1 = mean|a - G_BA(G_AB(a))| = {L_cyc_good:.6f}")

print("\n--- Poorly-trained CycleGAN (quality = 0.4) ---")
b_bad  = G_AB(a, 0.4)
a_bad  = G_BA(b_bad, 0.4)
print("Cycled back:"); print(np.round(a_bad, 3))
L_cyc_bad = np.abs(a - a_bad).mean()
print(f"Cycle-consistency loss  L1 = {L_cyc_bad:.6f}")

print(f"\nLoss comparison: good {L_cyc_good:.6f}  vs  bad {L_cyc_bad:.6f}")
print("The cycle loss forces the two generators to be near-inverses of each")
print("other, which is what prevents the network from ignoring the input and")
print("producing an arbitrary image in the target domain.\n")

# ---- Full CycleGAN objective ----
print("=== Full CycleGAN objective ===\n")
L_gan_AB, L_gan_BA, lam = 0.412, 0.389, 10.0
L_cycle = L_cyc_good + 0.0031
total = L_gan_AB + L_gan_BA + lam * L_cycle
print(f"  Adversarial loss  G_AB / D_B      : {L_gan_AB:.4f}")
print(f"  Adversarial loss  G_BA / D_A      : {L_gan_BA:.4f}")
print(f"  Cycle-consistency loss (both ways): {L_cycle:.4f}")
print(f"  lambda (cycle weight)             : {lam}")
print(f"  TOTAL = {L_gan_AB:.4f} + {L_gan_BA:.4f} + {lam}*{L_cycle:.4f} = {total:.4f}")

print("\n=== SuperstarGAN ===")
print("Combines ACGAN's separate classifier with CycleGAN's cycle loss.")
print("Reported FID on 10000 CelebA images (lower is better):")
for model, fid in [("SuperstarGAN", 22.7), ("StarGAN", 27.7),
                   ("SuperstarGAN without DA", 29.1),
                   ("StarGAN + SN + Hinge Loss", 34.0), ("StarGAN + DA", 38.3)]:
    print(f"  {model:<28} FID = {fid}")
print("  (DA = Data Augmentation, SN = Spectral Normalization)")
