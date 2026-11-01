"""
Chapter 23, Example 1: Vanilla RNN Forward Pass from Scratch
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This builds a vanilla RNN with the three weight matrices U, W, V and runs a length-4
sequence through it. You can watch the hidden state h_t evolve at each timestep - it
depends on the current input and the previous hidden state, which is the network's
memory. The same weights are reused at every timestep (weight sharing in time).

Run it:           python ex1_vanilla_rnn_forward_pass_from_scratch.py
Expected output:  expected_output/ex1_vanilla_rnn_forward_pass_from_scratch.txt
"""
# Chapter 23 - Example 1: Vanilla RNN forward pass from scratch
import numpy as np
np.random.seed(1)

def tanh(x): return np.tanh(x)

print("=== Vanilla RNN Forward Pass (from scratch) ===\n")
print("Architecture: input -> hidden (recurrent) -> output")
print("  h_t = tanh(U x_t + W h_(t-1) + b_h)")
print("  y_t = V h_t + b_o\n")

# Dimensions
d = 3   # input size
J = 4   # hidden size
K = 2   # output size

# Weight matrices (the 3 sets that define an RNN)
U = np.random.randn(J, d) * 0.5   # input -> hidden
W = np.random.randn(J, J) * 0.5   # hidden -> hidden (recurrent)
V = np.random.randn(K, J) * 0.5   # hidden -> output
b_h = np.zeros(J)
b_o = np.zeros(K)

print(f"U (input->hidden) : {U.shape}")
print(f"W (recurrent)     : {W.shape}")
print(f"V (hidden->output): {V.shape}\n")

# An input sequence of length T=4 (e.g., 4 words, each a 3-dim vector)
sequence = [np.array([1.0, 0.5, -0.5]),
            np.array([0.2, -1.0, 0.3]),
            np.array([-0.5, 0.8, 1.0]),
            np.array([0.9, 0.1, -0.2])]

print("Processing a sequence of length T=4:\n")
h = np.zeros(J)   # initial hidden state h_0
print(f"Initial hidden state h_0: {np.round(h, 3)}\n")

for t, x in enumerate(sequence, 1):
    h = tanh(U @ x + W @ h + b_h)
    y = V @ h + b_o
    print(f"t={t}: x_t={np.round(x,2)}")
    print(f"     h_{t} = {np.round(h, 4)}")
    print(f"     y_{t} = {np.round(y, 4)}")
    print()

print("Key insight: h_t depends on x_t AND h_(t-1), so the hidden state")
print("carries 'memory' of ALL previous inputs in the sequence.")
print("The SAME weights U, W, V are reused at every timestep (weight sharing in time).")
