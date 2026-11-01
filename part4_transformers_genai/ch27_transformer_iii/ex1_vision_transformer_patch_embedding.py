"""
Chapter 27, Example 1: Vision Transformer Patch Embedding
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This performs the four steps that turn an image into a sequence: split into patches,
flatten and linearly project them, prepend the learnable [CLS] token whose final output
is used for classification, and add position embeddings. The table at the end shows the
trade-off in patch size - smaller patches give more tokens and finer detail at higher
computational cost.

Run it:           python ex1_vision_transformer_patch_embedding.py
Expected output:  expected_output/ex1_vision_transformer_patch_embedding.txt
"""
# Chapter 27 - Example 1: Vision Transformer (ViT) patch embedding
import numpy as np
np.random.seed(0)

print("=== Vision Transformer (ViT): 'An image is worth 16x16 words' ===\n")
print("Idea: split the image into fixed-size patches, flatten each patch,")
print("linearly project it -> now the image IS a sequence, so a Transformer")
print("encoder can process it exactly like a sentence.\n")

H = W = 32          # image size
C = 3               # channels (RGB)
P = 8               # patch size
d = 16              # embedding dimension

image = np.random.rand(H, W, C)
print(f"Input image : {H}x{W}x{C}")
print(f"Patch size  : {P}x{P}")

n_patches = (H // P) * (W // P)
patch_dim = P * P * C
print(f"Number of patches N = (H/P) x (W/P) = {H//P} x {W//P} = {n_patches}")
print(f"Flattened patch dimension = P*P*C = {P}*{P}*{C} = {patch_dim}\n")

# 1. Split into patches and flatten
patches = []
for i in range(0, H, P):
    for j in range(0, W, P):
        patches.append(image[i:i+P, j:j+P, :].reshape(-1))
patches = np.array(patches)
print(f"Step 1 - patch matrix shape: {patches.shape}  (N x patch_dim)")

# 2. Linear projection to d dimensions
E = np.random.randn(patch_dim, d) * 0.05
embeddings = patches @ E
print(f"Step 2 - after linear projection E {E.shape}: {embeddings.shape}  (N x d)")

# 3. Prepend the learnable [CLS] token
cls_token = np.random.randn(1, d) * 0.05
seq = np.vstack([cls_token, embeddings])
print(f"Step 3 - prepend [CLS] token: {seq.shape}  (N+1 x d)")

# 4. Add position encodings
pos = np.random.randn(*seq.shape) * 0.05
seq = seq + pos
print(f"Step 4 - add position embeddings: {seq.shape}")

print(f"\nFinal sequence fed to the Transformer encoder: {seq.shape[0]} tokens of dim {d}")
print(f"  token 0      = [CLS]  (its final output is used for classification)")
print(f"  tokens 1..{n_patches} = image patches\n")
print(f"First 6 dims of [CLS] token : {np.round(seq[0][:6], 4)}")
print(f"First 6 dims of patch 1     : {np.round(seq[1][:6], 4)}")

print("\n=== Sequence length vs patch size ===\n")
print(f"{'Patch size':>11} {'#Patches':>10} {'Patch dim':>11}  {'Note'}")
print("-" * 62)
for p in [4, 8, 16]:
    n = (H//p) * (W//p)
    note = "more tokens = finer detail, slower" if p == 4 else (
           "fewer tokens = faster, coarser" if p == 16 else "balanced")
    print(f"{p:>7}x{p:<3} {n:>10} {p*p*C:>11}  {note}")

print("\nViT for image captioning: ViT encoder represents the image, then a")
print("Transformer decoder generates the caption word by word.")
