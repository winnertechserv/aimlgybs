"""
Chapter 22, Example 1: Residual Connections vs Vanishing Gradients
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This simulates backpropagation through a 20-layer network. In a plain network the
gradient is multiplied by a small factor at each layer and vanishes to ~1e-19 by the
first layer. With residual (skip) connections the gradient has a '+1' highway and stays
healthy. This is exactly why ResNet can train networks 50-152 layers deep.

Run it:           python ex1_residual_connections_vs_vanishing_gradients.py
Expected output:  expected_output/ex1_residual_connections_vs_vanishing_gradients.txt
"""
# Chapter 22 - Example 1: Residual connections fix vanishing gradients
import numpy as np
np.random.seed(0)

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_deriv(x): return sigmoid(x)*(1-sigmoid(x))

# Simulate gradient flow backward through a deep network
# Plain net: gradient multiplies by (weight * activation_deriv) each layer
# Residual net: gradient flows through identity skip (+1) so it doesn't vanish

n_layers = 20
np.random.seed(0)
weights = np.random.uniform(0.3, 0.9, n_layers)   # typical small weights
acts    = np.random.uniform(0.1, 0.25, n_layers)  # sigmoid derivative range

print("=== Gradient Flow Through a 20-Layer Network ===\n")
print("Plain network: grad_(L-1) = grad_L * w * sigmoid'(a)")
print("Residual net : grad_(L-1) = grad_L * (1 + w * sigmoid'(a))  <- the +1 is the skip\n")

grad_plain = 1.0
grad_res   = 1.0
print(f"{'Layer':>6} {'Plain grad':>14} {'Residual grad':>16}")
print("-" * 40)
for L in range(n_layers, 0, -1):
    factor = weights[L-1] * acts[L-1]
    grad_plain *= factor
    grad_res   *= (1 + factor)
    if L % 4 == 0 or L == 1:
        print(f"{L:>6} {grad_plain:>14.2e} {grad_res:>16.2e}")

print(f"\nAfter backprop through all 20 layers:")
print(f"  Plain network gradient   : {grad_plain:.2e}  (vanished to near-zero!)")
print(f"  Residual network gradient: {grad_res:.2e}  (healthy, training continues)")
print(f"\nRatio: residual gradient is {grad_res/grad_plain:.2e}x larger")
print("This is why ResNet can train 50, 101, even 152 layers deep.")
