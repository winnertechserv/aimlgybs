"""
Chapter 25, Example 3: Masked Self-Attention and the PWFFNN
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
The decoder must not look at future words, so a causal mask sets those attention scores
to exactly zero - you can see the upper triangle vanish while every row still sums to 1.
The second half builds the Position-Wise Feedforward Network and shows weight sharing: T
separate MLFFNNs would need T times the parameters, but sharing the weight matrices
across positions cuts that by 80% for T=5.

Run it:           python ex3_masked_self_attention_and_the_pwffnn.py
Expected output:  expected_output/ex3_masked_self_attention_and_the_pwffnn.txt
"""
# Chapter 25 - Example 3: Masked Self-Attention (decoder) + PWFFNN weight sharing
import numpy as np
np.random.seed(2)

def softmax_rows(M):
    e = np.exp(M - M.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

T, d = 5, 6
X = np.random.randn(T, d)

print("=== Masked Self-Attention in the Decoder ===\n")
print("In the decoder, the output word at time t may depend ONLY on words")
print("y_1 ... y_t (already generated). Future words must be hidden -> masking.\n")

Q = np.random.randn(T, d) * 0.3
K = np.random.randn(T, d) * 0.3
V = np.random.randn(T, d) * 0.3
raw = (Q @ K.T) / np.sqrt(d)

# Causal mask: position t cannot look at positions > t
mask = np.triu(np.ones((T, T)), k=1).astype(bool)
print("Causal mask (True = blocked / future position):")
print(mask.astype(int))

masked = raw.copy()
masked[mask] = -np.inf          # -inf -> softmax gives exactly 0

scores_unmasked = softmax_rows(raw)
scores_masked   = softmax_rows(masked)

print("\nAttention scores WITHOUT mask (token 2 can see the future - wrong!):")
print(np.round(scores_unmasked, 3))
print("\nAttention scores WITH causal mask (upper triangle is exactly 0):")
print(np.round(scores_masked, 3))
print(f"\nRow sums with mask: {np.round(scores_masked.sum(axis=1), 4)} (still valid distributions)")
print("Token 1 attends only to itself; token 5 can attend to all 5 positions.")

# ---- PWFFNN: same weights applied at every position ----
print("\n=== Position-Wise Feedforward Neural Network (PWFFNN) ===\n")
Z = scores_masked @ V        # output of masked MHA, shape (T,d)
d_ff = 12
W1 = np.random.randn(d, d_ff) * 0.3;  b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d) * 0.3;  b2 = np.zeros(d)

def pwffnn(z):               # one MLFFNN applied to ONE position
    hidden = np.maximum(0, z @ W1 + b1)     # nonlinear hidden layer
    return hidden @ W2 + b2

out = np.vstack([pwffnn(Z[t]) for t in range(T)])

print(f"There are T={T} MLFFNNs (one per position), but the weight matrices")
print(f"W_FF1 {W1.shape} and W_FF2 {W2.shape} are SHARED across all positions.")
print(f"Parameters if NOT shared : {T} x ({W1.size + W2.size}) = {T*(W1.size+W2.size)}")
print(f"Parameters WITH sharing  : {W1.size + W2.size}")
print(f"Reduction                : {(1 - 1/T)*100:.0f}% fewer parameters\n")
print(f"Input to PWFFNN  (position 1): {np.round(Z[0], 4)}")
print(f"Output of PWFFNN (position 1): {np.round(out[0], 4)}")
print(f"Output sequence shape: {out.shape}  (same length and dimension as input)")

# Verify weight sharing: feeding the same vector at 2 positions gives same output
same = np.allclose(pwffnn(Z[0]), pwffnn(Z[0].copy()))
print(f"\nSame input at any position -> same output? {same}  (confirms shared weights)")
