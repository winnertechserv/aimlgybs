"""
Chapter 25, Example 2: Self-Attention with Multi-Head Attention
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Here queries, keys and values are all generated from the SAME sequence using W_Q, W_K
and W_V - that is what makes it self-attention. We first run single-head attention, then
multi-head attention with h=4 heads. Notice each head produces a different attention
pattern for the same token, and that concatenating h heads of dimension l = d/h
reconstructs a d-dimensional vector, which W_o then transforms into the output.

Run it:           python ex2_self_attention_with_multi_head_attention.py
Expected output:  expected_output/ex2_self_attention_with_multi_head_attention.txt
"""
# Chapter 25 - Example 2: Self-Attention with Multi-Head Attention (MHA)
import numpy as np
np.random.seed(1)

def softmax_rows(M):
    e = np.exp(M - M.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# Sequence X: T d-dimensional feature vectors (e.g. word embeddings)
T, d, h = 4, 8, 4          # 4 tokens, model dim 8, 4 heads
l = d // h                 # per-head dim  l = d/h
X = np.round(np.random.randn(T, d), 2)

print("=== Self-Attention: Single-Head vs Multi-Head ===\n")
print(f"Sequence X: T={T} tokens, model dimension d={d}")
print(f"Heads h={h}  ->  per-head dimension l = d/h = {l}\n")

def head_attention(X, WQ, WK, WV):
    """One attention head. W*: (l x d). Returns context matrix (T x l)."""
    Q = X @ WQ.T          # (T,l)  query vectors  q~_it = W_i^Q x_t
    K = X @ WK.T          # (T,l)  key vectors
    V = X @ WV.T          # (T,l)  value vectors
    scores = softmax_rows((Q @ K.T) / np.sqrt(l))   # (T,T)
    return scores, scores @ V

# ---- Single-head attention (one set of transformation matrices) ----
WQ = np.random.randn(l, d) * 0.3
WK = np.random.randn(l, d) * 0.3
WV = np.random.randn(l, d) * 0.3
scores1, C1 = head_attention(X, WQ, WK, WV)

print("--- Single-Head Attention ---")
print(f"Transformation matrices W_Q, W_K, W_V each of size l x d = {l}x{d}")
print("Attention score matrix (row t = how token t attends to every token):")
print(np.round(scores1, 3))
print(f"Row sums: {np.round(scores1.sum(axis=1), 4)}  (each row sums to 1)")
print(f"Context matrix C~ shape: {C1.shape}  (l x T transposed -> T x l)\n")

# ---- Multi-head attention ----
print("--- Multi-Head Attention ---")
heads, all_scores = [], []
for i in range(h):
    WQi = np.random.randn(l, d) * 0.3
    WKi = np.random.randn(l, d) * 0.3
    WVi = np.random.randn(l, d) * 0.3
    s_i, C_i = head_attention(X, WQi, WKi, WVi)
    heads.append(C_i); all_scores.append(s_i)

print("Each head learns a DIFFERENT attention pattern for token 1:")
for i, s in enumerate(all_scores, 1):
    print(f"  Head {i}: {np.round(s[0], 3)}")

C_concat = np.concatenate(heads, axis=1)    # (T, h*l) = (T, d)
Wo = np.random.randn(d, d) * 0.3            # d x d output matrix
Z  = C_concat @ Wo.T                        # output sequence Z

print(f"\nConcatenated context: {h} heads x {l} dims = {C_concat.shape[1]} = d")
print(f"  c_t = concat(c_1t, ..., c_{h}t)  ->  shape {C_concat.shape}")
print(f"Output z_t = W_o c_t                ->  shape {Z.shape}")
print(f"\nOutput sequence Z (first 2 tokens):")
print(np.round(Z[:2], 4))

print("\nWhy multiple heads? Each head uses its own W_Q/W_K/W_V, so different")
print("heads can capture different relationships (syntax, coreference, etc.)")
print("in parallel. Their outputs are concatenated and mixed by W_o.")
