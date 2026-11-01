"""
Chapter 22, Example 2: Backpropagation Through Pooling Layers
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Pooling layers have no learnable weights, but gradients still must flow back through
them. This example shows the fixed routing weights: max pooling sends the whole gradient
only to the element that 'won' the max (a 0/1 mask), while mean pooling spreads the
gradient equally (1/F^2) across the region. This is the rule used when training a CNN.

Run it:           python ex2_backpropagation_through_pooling_layers.py
Expected output:  expected_output/ex2_backpropagation_through_pooling_layers.txt
"""
# Chapter 22 - Example 2: Max-pool vs Mean-pool BACKWARD pass (gradient routing)
import numpy as np

print("=== Backpropagation Through Pooling Layers ===\n")
print("Forward: pooling reduces a region to one value.")
print("Backward: the upstream gradient must be routed back to the input region.\n")

# A 2x2 region in the feature map
region = np.array([[0.8, -0.1],
                   [-1.3, 1.5]])
upstream_grad = 2.0  # gradient flowing back from next layer

print("Input region (2x2):")
print(region)
print(f"Upstream gradient arriving at this pooled node: {upstream_grad}\n")

# --- MAX POOLING backward ---
print("--- Max Pooling ---")
max_val = region.max()
mask = (region == max_val).astype(float)
print(f"Forward output (max): {max_val}")
print("Backward weights (1 only at the max location, 0 elsewhere):")
print(mask.astype(int))
grad_max = mask * upstream_grad
print("Gradient routed to input region:")
print(grad_max)
print("Only the neuron that 'won' the max receives the gradient.\n")

# --- MEAN POOLING backward ---
print("--- Mean Pooling ---")
mean_val = region.mean()
F2 = region.size  # F*F = 4
print(f"Forward output (mean): {mean_val:.3f}")
print(f"Backward weights (1/F^2 = 1/{F2} = {1/F2} at every location):")
print(np.full(region.shape, 1/F2))
grad_mean = np.full(region.shape, 1/F2) * upstream_grad
print("Gradient routed to input region:")
print(grad_mean)
print("Every neuron in the region shares the gradient equally.\n")

print("=== Summary ===")
print("Max pooling : gradient goes ONLY to the max element (sparse, 0/1 weights)")
print("Mean pooling: gradient is SPREAD evenly (each gets 1/F^2)")
print("Pooling layers have no learnable weights - these routing weights are fixed.")
