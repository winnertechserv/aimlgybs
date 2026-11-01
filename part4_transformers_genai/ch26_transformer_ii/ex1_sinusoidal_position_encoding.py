"""
Chapter 26, Example 1: Sinusoidal Position Encoding
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This builds the position encoding matrix using the exact formula, with sin for even
indices and cos for odd indices. The output shows that every position gets a distinct
vector, that the same word embedding becomes different at different positions, and that
the dot product between position vectors decays with distance - which is how the model
infers relative order.

Run it:           python ex1_sinusoidal_position_encoding.py
Expected output:  expected_output/ex1_sinusoidal_position_encoding.txt
"""
# Chapter 26 - Example 1: Sinusoidal Position Encoding
import numpy as np
np.set_printoptions(suppress=True)

def position_encoding(T, d):
    """p_t[i] = sin(w_k * t) if i=2k ; cos(w_k * t) if i=2k+1, w_k = 1/10000^(2k/d)"""
    P = np.zeros((T, d))
    for t in range(T):
        for i in range(d):
            k  = i // 2
            wk = 1.0 / (10000 ** (2 * k / d))
            P[t, i] = np.sin(wk * t) if i % 2 == 0 else np.cos(wk * t)
    return P

print("=== Sinusoidal Position Encoding ===\n")
print("Why needed? Attention has NO notion of order. Consider:")
print('  "The food was good, not bad at all"')
print('  "The food was bad, not good at all"')
print("Same words, opposite meaning -> position information is essential.\n")

T, d = 6, 8
P = position_encoding(T, d)
print(f"Position encoding matrix p_t for T={T} positions, d={d}:")
print(np.round(P, 4))

print("\nEvery position gets a DIFFERENT vector:")
for t in range(3):
    print(f"  p_{t} = {np.round(P[t], 4)}")

print("\nCombine with word embedding:  x_hat_t = x_t + p_t")
np.random.seed(0)
x = np.round(np.random.randn(d), 3)
print(f"  word embedding x  (same word at any position): {x}")
print(f"  x_hat at position 0 = {np.round(x + P[0], 3)}")
print(f"  x_hat at position 3 = {np.round(x + P[3], 3)}")
print("  -> the SAME word gets different embeddings at different positions.")

print("\n=== Property: similarity decays with distance ===")
print("Dot product between position 0 and position t:")
for t in range(T):
    print(f"  p_0 . p_{t} = {P[0] @ P[t]:+.4f}")
print("Nearby positions have higher similarity -> the model can infer relative order.")

print("\n=== Why sinusoidal (not learned integers)? ===")
print("  * Values stay bounded in [-1, 1] regardless of sequence length")
print("  * Generalizes to sequences longer than those seen in training")
print("  * Relative position is expressible as a linear function of p_t")
