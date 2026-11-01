"""
Chapter 23, Example 2: LSTM Cell from Scratch
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This computes one full timestep of an LSTM cell: the forget, input, and output gates
(sigmoid), the candidate state (tanh), the additive cell-state update, and the hidden-
state read-out. The additive cell-state update is what lets gradients survive over long
sequences. The parameter count confirms an LSTM has ~4x the weights of a vanilla RNN
layer.

Run it:           python ex2_lstm_cell_from_scratch.py
Expected output:  expected_output/ex2_lstm_cell_from_scratch.txt
"""
# Chapter 23 - Example 2: LSTM cell from scratch (one timestep, all gates)
import numpy as np
np.random.seed(2)

def sigmoid(x): return 1/(1+np.exp(-x))

print("=== LSTM Cell - One Timestep, All Gates ===\n")
print("An LSTM has 3 gates + a cell state, designed to avoid vanishing gradients:")
print("  Forget gate f_t : what to erase from cell state")
print("  Input gate  i_t : what new info to write")
print("  Output gate o_t : what to read out as the hidden state")
print("  Cell state  s_t : the long-term memory conveyor belt\n")

d = 3   # input size
J = 4   # hidden/cell size

# Each gate has its own U (input) and W (recurrent) weight matrices
def rand(*shape): return np.random.randn(*shape) * 0.3
U_i, W_i, b_i = rand(J,d), rand(J,J), np.zeros(J)   # input gate
U_f, W_f, b_f = rand(J,d), rand(J,J), np.zeros(J)   # forget gate
U_o, W_o, b_o = rand(J,d), rand(J,J), np.zeros(J)   # output gate
U_c, W_c, b_c = rand(J,d), rand(J,J), np.zeros(J)   # candidate (temporary state)

# Previous states
x_t   = np.array([0.5, -0.3, 0.8])   # current input
h_prev = np.array([0.1, -0.2, 0.05, 0.15])   # previous hidden state
s_prev = np.array([0.2, 0.1, -0.1, 0.3])     # previous cell state

print(f"Current input  x_t   : {x_t}")
print(f"Previous hidden h_(t-1): {h_prev}")
print(f"Previous cell   s_(t-1): {s_prev}\n")

# Gate computations
i_t = sigmoid(U_i @ x_t + W_i @ h_prev + b_i)
f_t = sigmoid(U_f @ x_t + W_f @ h_prev + b_f)
o_t = sigmoid(U_o @ x_t + W_o @ h_prev + b_o)
s_tilde = np.tanh(U_c @ x_t + W_c @ h_prev + b_c)   # candidate

print("Step 1 - Gate activations (sigmoid -> values in 0..1):")
print(f"  Input gate  i_t = {np.round(i_t, 3)}")
print(f"  Forget gate f_t = {np.round(f_t, 3)}")
print(f"  Output gate o_t = {np.round(o_t, 3)}")
print(f"  Candidate ~s_t  = {np.round(s_tilde, 3)}  (tanh, -1..1)\n")

# Cell state update: selective forget + selective write
s_t = f_t * s_prev + i_t * s_tilde
print("Step 2 - Cell state update (selective forget + selective write):")
print(f"  s_t = f_t (*) s_(t-1) + i_t (*) ~s_t")
print(f"      = {np.round(f_t*s_prev,3)} + {np.round(i_t*s_tilde,3)}")
print(f"      = {np.round(s_t, 4)}\n")

# Hidden state: selective read
h_t = o_t * np.tanh(s_t)
print("Step 3 - Hidden state (selective read):")
print(f"  h_t = o_t (*) tanh(s_t) = {np.round(h_t, 4)}\n")

print("Why no vanishing gradient? The cell state s_t updates ADDITIVELY")
print("(f_t * s_(t-1) + ...), so gradients flow back through time without")
print("being repeatedly multiplied down to zero - unlike a vanilla RNN.")

# Param count comparison
print(f"\nParameter count: LSTM has 4 weight sets (i,f,o,candidate)")
print(f"  = 4x the parameters of a vanilla RNN hidden layer")
print(f"  Vanilla RNN: U({J}x{d}) + W({J}x{J}) = {J*d + J*J} weights")
print(f"  LSTM       : 4 x ({J*d + J*J}) = {4*(J*d + J*J)} weights")
