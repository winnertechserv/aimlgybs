"""
Chapter 25, Example 1: Scaled Dot-Product Attention from Scratch
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This builds the core attention computation exactly as defined in the overview. We build
Query, Key and Value vectors, compute the scaled dot products alpha_tm =
<q_t,k_m>/sqrt(d), apply softmax to get attention scores, and form the context vector as
the weighted sum of Value vectors. The final section shows concretely why the 1/sqrt(d)
scaling matters: without it, large dot products saturate the softmax and gradients
vanish.

Run it:           python ex1_scaled_dot_product_attention_from_scratch.py
Expected output:  expected_output/ex1_scaled_dot_product_attention_from_scratch.txt
"""
# Chapter 25 - Example 1: Scaled Dot-Product Attention (SDPA) from scratch
import numpy as np
np.random.seed(0)

def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()

def sdpa(q, K, V, d):
    """q: query (d,) | K: keys (T,d) | V: values (T,d)"""
    alpha = (K @ q) / np.sqrt(d)      # scaled dot products
    a     = softmax(alpha)            # attention scores
    c     = a @ V                     # context vector
    return alpha, a, c

d, T = 4, 5
K = np.round(np.random.randn(T, d), 2)   # Key vectors
V = np.round(np.random.randn(T, d), 2)   # Value vectors
q = np.round(np.random.randn(d), 2)      # Query vector for position t

print("=== Scaled Dot-Product Attention ===\n")
print("Terminology borrowed from regression: Query, Key, Value\n")
print(f"Query q_t          : {q}")
print(f"Key matrix K  ({T}x{d}):\n{K}")
print(f"Value matrix V ({T}x{d}):\n{V}\n")

alpha, a, c = sdpa(q, K, V, d)

print("Step 1: scaled dot product  alpha_tm = <q_t, k_m> / sqrt(d)")
for m in range(T):
    print(f"  alpha_t{m+1} = {K[m]} . q / sqrt({d}) = {alpha[m]:+.4f}")

print("\nStep 2: attention scores  a_tm = softmax(alpha_tm)")
for m in range(T):
    bar = "#" * int(a[m] * 40)
    print(f"  a_t{m+1} = {a[m]:.4f}  {bar}")
print(f"  Sum of attention scores = {a.sum():.4f}  (softmax always sums to 1)")

print("\nStep 3: context vector  c_t = sum_m a_tm * v_m")
print(f"  c_t = {np.round(c, 4)}")

print("\nInterpretation:")
top = int(np.argmax(a))
print(f"  Key {top+1} receives the highest attention ({a[top]:.4f}),")
print(f"  so the context vector is pulled most towards Value vector {top+1}.")
print("\nWhy divide by sqrt(d)? Without scaling, large dot products push softmax")
print("into a saturated region where gradients vanish during training.")

# Show the saturation effect
print("\n=== Effect of the 1/sqrt(d) scaling ===")
big = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
print(f"  Unscaled logits {big} -> softmax {np.round(softmax(big), 4)}")
print(f"  Scaled by sqrt({d})            -> softmax {np.round(softmax(big/np.sqrt(d)), 4)}")
print("  Scaling keeps the distribution smoother (non-saturated gradients).")
