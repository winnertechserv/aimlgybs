"""
Chapter 21, Example 3: Feature-Map Size & MLFFNN vs CNN Parameters
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This applies the feature-map size formula to several configurations (including padding)
and then drives home why CNNs are efficient: a fully-connected layer on 16 pixels needs
64 weights, while the equivalent convolution needs just the shared kernel weights - a
~94% reduction. The LeNet-5 example shows ~5 million dense weights collapsing to ~156
with weight sharing.

Run it:           python ex3_feature_map_size_mlffnn_vs_cnn_parameters.py
Expected output:  expected_output/ex3_feature_map_size_mlffnn_vs_cnn_parameters.txt
"""
# Chapter 21 - Example 3: Feature-map size + MLFFNN vs CNN parameter comparison
import numpy as np

def feature_map_size(W1, H1, F, S, P=0):
    W2 = (W1 - F + 2*P)//S + 1
    H2 = (H1 - F + 2*P)//S + 1
    return W2, H2

print("=== Feature-Map Size Formula: W2 = (W1 - F + 2P)/S + 1 ===\n")
configs = [
    (4, 4, 2, 2, 0),
    (4, 4, 2, 1, 0),
    (32, 32, 3, 2, 0),
    (32, 32, 4, 3, 0),
    (4, 4, 2, 2, 2),   # with padding
]
print(f"{'Input':>8} {'Filter':>7} {'Stride':>7} {'Pad':>5} {'Output':>8}")
print("-" * 40)
for W1, H1, F, S, P in configs:
    W2, H2 = feature_map_size(W1, H1, F, S, P)
    print(f"{W1}x{H1:<5} {F}x{F:<5} {S:>5} {P:>5} {W2}x{H2}")

print("\n=== MLFFNN vs CNN: Parameter Count (16-pixel input, 4 hidden units) ===\n")
# MLFFNN: fully connected — every input connects to every hidden node
n_input  = 16   # 4x4 image flattened
n_hidden = 4
mlffnn_weights = n_input * n_hidden
print(f"MLFFNN (fully connected):")
print(f"  Connections = {n_input} inputs x {n_hidden} hidden = {mlffnn_weights} weights")

# CNN: 2x2 kernel, stride 2 -> 4 output nodes, but SHARED weights
kernel_size = 2 * 2   # 2x2 kernel
print(f"\nCNN (convolution, 2x2 kernel, stride 2):")
print(f"  4 hidden nodes each use the SAME 2x2 kernel (weight sharing)")
print(f"  Unique weights = {kernel_size} (one shared kernel) + 1 bias = {kernel_size + 1}")
print(f"\nParameter reduction: {mlffnn_weights} -> {kernel_size} "
      f"({(1 - kernel_size/mlffnn_weights)*100:.1f}% fewer weights)")

# LeNet-5 style realistic example
print("\n=== Real example: LeNet-5 first conv layer ===")
print("Input 32x32x1, Conv layer CL1: 6 kernels of 5x5, stride 1")
n_kernels = 6; ksz = 5*5
conv_params = n_kernels * (ksz + 1)  # +1 bias per kernel
W2, H2 = feature_map_size(32, 32, 5, 1, 0)
print(f"  Output feature maps: {W2}x{H2}x{n_kernels}")
print(f"  Learnable parameters: {n_kernels} x (5x5 + 1 bias) = {conv_params}")
fc_equiv = (32*32) * (W2*H2*n_kernels)
print(f"  A fully-connected layer for the same output would need ~{fc_equiv:,} weights")
print(f"  CNN uses {conv_params} -- that's the power of weight sharing + sparse connections!")
