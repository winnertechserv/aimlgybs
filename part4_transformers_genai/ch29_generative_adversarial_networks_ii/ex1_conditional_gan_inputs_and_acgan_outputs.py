"""
Chapter 29, Example 1: Conditional GAN Inputs and ACGAN Outputs
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This shows exactly how the conditioning information C enters a conditional GAN: the
class label is one-hot encoded and concatenated with the noise vector, so the same z
with a different label produces a different image. The second half shows the ACGAN
discriminator's two outputs - the probability the image is real and the predicted class
label from the auxiliary classifier.

Run it:           python ex1_conditional_gan_inputs_and_acgan_outputs.py
Expected output:  expected_output/ex1_conditional_gan_inputs_and_acgan_outputs.txt
"""
# Chapter 29 - Example 1: Conditional GAN inputs + ACGAN dual outputs
import numpy as np
np.random.seed(0)
np.set_printoptions(suppress=True)

print("=== Conditional GAN (cGAN) ===\n")
print("Both G and D are conditioned on additional information C.")
print("C can be a class label, a source-domain image, a sketch, or text.\n")

n_classes, z_dim = 5, 6
classes = ["cat", "dog", "bird", "car", "tree"]

def one_hot(k, n):
    v = np.zeros(n); v[k] = 1.0; return v

# ---- Class-conditional GAN ----
print("--- Class-Conditional GAN: C = class label ---\n")
z = np.round(np.random.randn(z_dim), 2)
label = 2                                    # want to generate a "bird"
c = one_hot(label, n_classes)
g_input = np.concatenate([z, c])

print(f"Random noise vector z ({z_dim}-dim)      : {z}")
print(f"Class label            : {label} = '{classes[label]}'")
print(f"One-hot encoding C     : {c}")
print(f"Generator input [z ; C]: {g_input}   (dim {len(g_input)})")
print("\nThe SAME z with a DIFFERENT label produces a different image:")
for k in [0, 3]:
    gi = np.concatenate([z, one_hot(k, n_classes)])
    print(f"  label {k} ('{classes[k]:<5}') -> input {gi}")

print("\nDiscriminator also receives C, so it must judge:")
print("  'is this a REAL image AND does it match the given label?'\n")

# ---- ACGAN ----
print("--- Auxiliary Classifier GAN (ACGAN) ---\n")
print("The discriminator has TWO outputs:")
print("  1. probability the input image is real")
print("  2. the class label of the input image (auxiliary classifier C)\n")

def softmax(v):
    e = np.exp(v - v.max()); return e / e.sum()

for tag, p_real, logits in [
    ("real 'bird' image     ", 0.93, np.array([0.2, 0.4, 3.1, 0.1, 0.3])),
    ("synthetic 'bird' image", 0.21, np.array([0.3, 0.5, 2.2, 0.2, 0.4])),
]:
    probs = softmax(logits)
    pred = int(np.argmax(probs))
    print(f"  {tag}: D_real = {p_real:.2f} | class probs = {np.round(probs,3)}")
    print(f"  {'':<22}  predicted class = '{classes[pred]}' (correct: bird)")

print("\nACGAN loss has two parts:")
print("  L_source = log P(real | x)     -> adversarial term")
print("  L_class  = log P(class | x)    -> classification term")
print("  D maximises L_source + L_class ; G maximises L_class - L_source")

print("\n--- Other conditional variants ---")
print("  BigGAN          : class-conditional GAN on ImageNet, built from ResBlocks")
print("  pix2pix (cGAN)  : C = image from the source domain (image-to-image)")
print("  Sketch-to-photo : C = a sketch")
print("  Text-to-image   : C = representation of a text prompt")
