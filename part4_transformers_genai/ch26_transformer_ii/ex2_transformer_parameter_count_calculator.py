"""
Chapter 26, Example 2: Transformer Parameter-Count Calculator
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This turns the complexity formulas into working code: 3ldhN for each MHA group, 3d^2N
for the W_o matrices, 4d^2N for the PWFFNN sub-layers and dV for the decoder's linear
layer. Applying it to the original base configuration (d=512, h=8, N=6) gives about 44
million parameters, and the comparison table shows that doubling d roughly quadruples
the count while doubling N only doubles it.

Run it:           python ex2_transformer_parameter_count_calculator.py
Expected output:  expected_output/ex2_transformer_parameter_count_calculator.txt
"""
# Chapter 26 - Example 2: Transformer parameter-count calculator
print("=== Complexity of the Transformer Model ===\n")
print("Parameter counts:")
print("  Encoder self-attention MHA in N layers : 3*l*d*h*N")
print("  Decoder self-attention MHA in N layers : 3*l*d*h*N")
print("  Cross-attention MHA in N layers        : 3*l*d*h*N")
print("  W_o matrices in 3 MHA sub-layers       : 3*d^2*N")
print("  Two weight matrices in 2 PWFFNN        : 4*d^2*N")
print("  Linear layer in decoder                : d*V\n")

def transformer_params(d, h, N, V, verbose=True):
    l = d // h                      # per-head dimension
    enc_mha   = 3 * l * d * h * N
    dec_mha   = 3 * l * d * h * N
    cross_mha = 3 * l * d * h * N
    wo        = 3 * d * d * N
    pwffnn    = 4 * d * d * N
    linear    = d * V
    total = enc_mha + dec_mha + cross_mha + wo + pwffnn + linear
    if verbose:
        print(f"Configuration: d={d}, h={h} heads (l={l}), N={N} layers, vocab V={V}")
        print(f"  Encoder self-attention MHA : {enc_mha:>12,}")
        print(f"  Decoder self-attention MHA : {dec_mha:>12,}")
        print(f"  Cross-attention MHA        : {cross_mha:>12,}")
        print(f"  W_o matrices               : {wo:>12,}")
        print(f"  PWFFNN weight matrices     : {pwffnn:>12,}")
        print(f"  Decoder linear layer       : {linear:>12,}")
        print(f"  {'TOTAL':<27}{total:>12,}\n")
    return total

# The original "Attention is all you need" base configuration
transformer_params(d=512, h=8, N=6, V=37000)

print("=== How parameters scale with each hyperparameter ===\n")
print(f"{'Config':<38} {'Parameters':>14}")
print("-" * 54)
for (d, h, N, V, label) in [
    (256, 4, 2, 10000, "small  (d=256, h=4,  N=2)"),
    (512, 8, 6, 37000, "base   (d=512, h=8,  N=6)"),
    (1024,16, 6, 37000, "large  (d=1024,h=16, N=6)"),
    (512, 8,12, 37000, "deep   (d=512, h=8,  N=12)"),
]:
    p = transformer_params(d, h, N, V, verbose=False)
    print(f"  {label:<36} {p:>14,}")

print("\nObservations:")
print("  * Doubling d roughly QUADRUPLES the parameters (terms are O(d^2)).")
print("  * Doubling N doubles the parameters (linear in depth).")
print("  * Changing h alone does not change the count, since l = d/h keeps l*h = d.")

print("\n=== Training the Transformer ===")
print("  * Trained end-to-end with error backpropagation, cross-entropy loss.")
print("  * Training data for translation: (source sentence, target sentence) pairs.")
print("  * Desired output at time t: one-hot vector of the t-th target word.")
print("  * Error is backpropagated through decoder then encoder layers, N, N-1, ..., 1.")
