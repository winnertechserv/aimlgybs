"""
Chapter 20, Example 3: Batch Normalization vs Layer Normalization — Side by Side
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
The final example clarifies the most commonly confused pair: Batch Normalization vs
Layer Normalization. Both normalize to mean=0, std=1 — but they normalize along
different axes. BatchNorm normalizes each neuron (column) across the mini-batch — so
neuron 1's output across all 3 examples becomes standardized. LayerNorm normalizes each
example (row) across its own neurons — so example 1's outputs across all 4 neurons
become standardized. The practical implication: BatchNorm needs a large batch to
estimate good statistics. LayerNorm works with batch size 1. That's why Transformers use
LayerNorm — they process sequences where batch size may be 1.

Run it:           python ex3_batch_normalization_vs_layer_normalization_si.py
Expected output:  expected_output/ex3_batch_normalization_vs_layer_normalization_si.txt
"""
import numpy as np

H = np.array([[1., 2., 3., 4.],
              [5., 6., 7., 8.],
              [2., 4., 6., 8.]])  # 3 examples x 4 neurons

print("Hidden layer output H (3 examples x 4 neurons):")
print(H)

print("\n=== Batch Normalization ===")
print("Normalize each COLUMN (neuron) across all 3 examples\n")
mu_bn  = H.mean(axis=0)      # shape (4,)
std_bn = H.std(axis=0) + 1e-8
H_bn   = (H - mu_bn) / std_bn
print(f"Column means: {mu_bn.round(4)}")
print(f"Column stds : {std_bn.round(4)}")
print("BN output:")
print(H_bn.round(4))
print(f"Verify: mean per column = {H_bn.mean(axis=0).round(4)}  (all ~0)")
print(f"        std  per column = {H_bn.std(axis=0).round(4)}  (all ~1)")

print("\n=== Layer Normalization ===")
print("Normalize each ROW (example) across its own 4 neurons\n")
mu_ln  = H.mean(axis=1, keepdims=True)   # shape (3,1)
std_ln = H.std(axis=1, keepdims=True) + 1e-8
H_ln   = (H - mu_ln) / std_ln
print(f"Row means: {mu_ln.flatten().round(4)}")
print(f"Row stds : {std_ln.flatten().round(4)}")
print("LN output:")
print(H_ln.round(4))
print(f"Verify: mean per row = {H_ln.mean(axis=1).round(4)}  (all ~0)")
print(f"        std  per row = {H_ln.std(axis=1).round(4)}  (all ~1)")

print("\n=== Summary ===")
print("  Batch Norm : normalizes across BATCH (columns) — needs large batch")
print("  Layer Norm : normalizes across FEATURES (rows) — works with batch=1")
print("  Use BatchNorm for CNNs and MLPs; LayerNorm for Transformers/RNNs")
