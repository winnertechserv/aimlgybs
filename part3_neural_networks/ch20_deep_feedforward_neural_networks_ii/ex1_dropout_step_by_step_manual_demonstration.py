"""
Chapter 20, Example 1: Dropout — Step-by-Step Manual Demonstration
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We build dropout from scratch to show exactly what happens during training and testing.
During training, a random binary mask is applied to the layer — zeroing out randomly
chosen nodes for each mini-batch. Each mini-batch gets a different thinned network. With
L=6 nodes, there are 2^6=64 possible thinned networks — we're implicitly training an
ensemble of 64 models in parallel! At test time, all nodes are active but weights are
multiplied by p_keep (the retention probability) to approximate the average of all 64
networks. We also show the key difference from DropConnect: Dropout zeros entire node
rows; DropConnect zeros individual weight connections.

Run it:           python ex1_dropout_step_by_step_manual_demonstration.py
Expected output:  expected_output/ex1_dropout_step_by_step_manual_demonstration.txt
"""
import numpy as np
np.random.seed(42)

L      = 6
p_keep = 0.5
layer  = np.random.randn(4, L)  # 4 examples x 6 nodes

print("=== Dropout: Training Phase ===\n")
print("Layer outputs BEFORE dropout (4 examples x 6 nodes):")
print(layer.round(4))
print()
for mb in range(1, 4):
    mask = (np.random.rand(L) > (1 - p_keep)).astype(float)
    out  = layer * mask
    dropped = [int(i) for i in np.where(mask == 0)[0]]
    print(f"Mini-batch {mb}: mask = {mask.astype(int)}  dropped nodes: {dropped}")
    print(f"  Output (example 1): {out[0].round(4)}")
    print(f"  Active nodes: {int(mask.sum())}/{L}  (thinned network)\n")

print("=== Test Phase: No Dropout, Scale Weights ===\n")
W = np.random.randn(L, 3)
print(f"At test time: all {L} nodes active BUT weights * p_keep={p_keep}")
print(f"  Original weight row 1: {W[0].round(4)}")
print(f"  Scaled   weight row 1: {(W[0]*p_keep).round(4)}")
print(f"  This approximates averaging over all 2^{L}={2**L} thinned networks\n")

print("=== Dropout vs DropConnect ===\n")
W_orig = np.array([[0.5,-0.3,0.8],[0.2,0.9,-0.4],[-0.6,0.1,0.7]])
print("Original W:"); print(W_orig)

W_drop = W_orig * np.array([1,0,1])[:,None]   # node 2 dropped (row zeroed)
print("\nDropout (node 2 dropped — full row 2 zeroed):")
print(W_drop)

dc_mask = np.array([[1,0,1],[1,1,0],[0,1,1]])
W_dc = W_orig * dc_mask
print("\nDropConnect (individual connections randomly zeroed):")
print(W_dc)
print("\nKey: Dropout removes NODES; DropConnect removes individual WEIGHTS.")
