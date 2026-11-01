"""
Chapter 18, Example 1: Backpropagation — Manual Forward & Backward Pass
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This is the most important code in the entire book. We build backpropagation completely
from scratch — no sklearn, no PyTorch, just numpy and math. The network has 2 inputs, 2
hidden tanh neurons, and 1 logistic output neuron. Given one training example (x=[0.5,
0.8], target=1.0), we trace the full forward pass (computing activations step by step),
then the full backward pass (computing all gradients using the chain rule). The final
check confirms the error decreased after the weight update — proving that one step of
gradient descent worked correctly.

Run it:           python ex1_backpropagation_manual_forward_backward_pass.py
Expected output:  expected_output/ex1_backpropagation_manual_forward_backward_pass.txt
"""
import numpy as np

np.random.seed(42)
x      = np.array([0.5, 0.8])
target = 1.0
eta    = 0.5

W1 = np.array([[0.15, 0.25],[0.20, 0.30]])  # hidden weights
b1 = np.array([0.35, 0.35])
W2 = np.array([[0.40, 0.45]])               # output weights
b2 = np.array([0.60])

sigmoid = lambda z: 1 / (1 + np.exp(-z))
tanh    = lambda z: np.tanh(z)

# ── FORWARD PASS ──
a_h = W1 @ x + b1       # hidden net inputs
s_h = tanh(a_h)         # hidden outputs (tanh)
a_o = W2 @ s_h + b2     # output net input
s_o = sigmoid(a_o)      # final output

E = 0.5 * (target - s_o[0])**2
print("=== FORWARD PASS ===")
print(f"Hidden net inputs a_h : {a_h.round(4)}")
print(f"Hidden outputs s_h    : {s_h.round(4)}")
print(f"Output net input a_o  : {a_o[0]:.4f}")
print(f"Network output s_o    : {s_o[0]:.4f}")
print(f"Target                : {target}")
print(f"Error E               : {E:.6f}")

# ── BACKWARD PASS ──
print("\n=== BACKWARD PASS ===")
delta_o = -(target - s_o[0]) * s_o[0] * (1 - s_o[0])
print(f"Output local gradient delta_o: {delta_o:.6f}")

dW2  = delta_o * s_h
W2_n = W2 - eta * dW2
b2_n = b2 - eta * delta_o
print(f"Updated output weights: {W2_n.round(6)}")

delta_h = delta_o * W2[0] * (1 - s_h**2)
print(f"Hidden local gradients delta_h: {delta_h.round(6)}")

dW1  = np.outer(delta_h, x)
W1_n = W1 - eta * dW1

# Verify: error should decrease
a_h2 = W1_n @ x + b1; s_h2 = tanh(a_h2)
s_o2 = sigmoid(W2_n @ s_h2 + b2_n)
E2   = 0.5 * (target - s_o2[0])**2
print(f"\nError BEFORE update: {E:.6f}")
print(f"Error AFTER  update: {E2:.6f}")
print(f"Error reduced by   : {E - E2:.6f}  <- gradient descent worked!")
