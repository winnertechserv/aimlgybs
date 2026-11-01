"""
Chapter 27, Example 2: CLIP Contrastive Image-Caption Matching
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This builds CLIP's training objective: encode images and captions into the same space,
compute the cosine similarity of every pair, then maximise the N correct pairs and
minimise the N^2 - N incorrect ones. The similarity matrix shows a strong diagonal,
retrieval is 100% accurate, and the contrastive loss is computed in both directions.

Run it:           python ex2_clip_contrastive_image_caption_matching.py
Expected output:  expected_output/ex2_clip_contrastive_image_caption_matching.txt
"""
# Chapter 27 - Example 2: CLIP-style contrastive image-caption matching
import numpy as np
np.random.seed(7)

print("=== CLIP: Contrastive Language-Image Pre-training ===\n")
print("Training objective:")
print("  1. Encode all images and all captions into the SAME m-dim space")
print("  2. Compute cosine similarity of every (image, caption) pair")
print("  3. MAXIMISE similarity of the N correct pairs")
print("     MINIMISE similarity of the N^2 - N incorrect pairs\n")

N, m = 4, 8
captions = ["a dog on a beach", "a plate of pasta",
            "a red sports car", "two children playing"]

# Simulate encoders that have already learned aligned representations
base = np.random.randn(N, m)
img_emb = base + 0.25 * np.random.randn(N, m)   # image encoder output
txt_emb = base + 0.25 * np.random.randn(N, m)   # text encoder output

def l2norm(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)

I, Tt = l2norm(img_emb), l2norm(txt_emb)
S = I @ Tt.T                     # cosine similarity matrix (N x N)

print(f"Similarity matrix S (rows = images, cols = captions):\n")
head = "            " + "".join(f"cap{j+1:<7}" for j in range(N))
print(head)
for i in range(N):
    row = "".join(f"{S[i,j]:+.3f}  " for j in range(N))
    print(f"  image{i+1}   {row}")

print(f"\nDiagonal (correct pairs)  : {np.round(np.diag(S), 3)}")
off = S[~np.eye(N, dtype=bool)]
print(f"Off-diagonal (wrong pairs): mean {off.mean():+.3f}")
print(f"Number of wrong pairs = N^2 - N = {N*N - N}")

# Retrieval accuracy
pred = S.argmax(axis=1)
acc = (pred == np.arange(N)).mean()
print(f"\nImage -> caption retrieval (pick argmax per row):")
for i in range(N):
    ok = "correct" if pred[i] == i else "WRONG"
    print(f"  image{i+1} -> '{captions[pred[i]]}'   [{ok}]")
print(f"Retrieval accuracy: {acc*100:.0f}%")

# Contrastive (InfoNCE) loss
tau = 0.07
def softmax_rows(M):
    e = np.exp(M - M.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
P = softmax_rows(S / tau)
loss_i2t = -np.mean(np.log(np.diag(P) + 1e-12))
Pt = softmax_rows((S.T) / tau)
loss_t2i = -np.mean(np.log(np.diag(Pt) + 1e-12))
print(f"\nContrastive loss (temperature tau={tau}):")
print(f"  image->text loss = {loss_i2t:.4f}")
print(f"  text->image loss = {loss_t2i:.4f}")
print(f"  total = (sum)/2  = {(loss_i2t + loss_t2i)/2:.4f}")

print("\nCLIP is used in DALL-E 2: the CLIP text encoder maps the caption into")
print("representation space, then a diffusion prior maps it to a CLIP image")
print("encoding, and reverse diffusion generates the final image.")
