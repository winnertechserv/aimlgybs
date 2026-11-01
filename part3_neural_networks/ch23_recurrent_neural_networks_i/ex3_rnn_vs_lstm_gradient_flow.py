"""
Chapter 23, Example 3: RNN vs LSTM Gradient Flow
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This traces the gradient backward through 30 timesteps. In a vanilla sigmoid RNN each
step multiplies the gradient by a factor below 1, so it vanishes to ~1e-21. In an LSTM
the cell-state highway keeps the gradient near its original size. This is the concrete
reason LSTMs learn long-range dependencies that plain RNNs cannot.

Run it:           python ex3_rnn_vs_lstm_gradient_flow.py
Expected output:  expected_output/ex3_rnn_vs_lstm_gradient_flow.txt
"""
# Chapter 23 - Example 3: RNN vs LSTM gradient flow over a long sequence
import numpy as np
np.random.seed(3)

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_deriv(x): return sigmoid(x)*(1-sigmoid(x))

print("=== Why Vanilla RNNs Forget: Gradient Flow Over Time ===\n")
print("During BPTT (Backprop Through Time), the gradient at the first timestep")
print("depends on a product of factors across ALL timesteps.\n")

T = 30  # sequence length

# Vanilla RNN: gradient multiplies by (W * sigmoid') each step back
# Typical sigmoid derivative <= 0.25, recurrent weight ~ moderate
W_rnn = 0.8
print(f"Vanilla RNN (recurrent weight W={W_rnn}, sigmoid neurons):")
grad = 1.0
trace_rnn = []
for t in range(T):
    deriv = 0.25  # max sigmoid derivative
    grad *= W_rnn * deriv
    trace_rnn.append(grad)
print(f"  Gradient reaching t=1 after {T} steps: {grad:.3e}")
print(f"  (each step multiplies by W*sigmoid' = {W_rnn*0.25} < 1, so it vanishes)\n")

# LSTM: cell-state path has near-1 multiplier (forget gate ~ 1)
print("LSTM (cell-state highway, forget gate f_t ~ 0.95):")
grad_lstm = 1.0
f_gate = 0.95
trace_lstm = []
for t in range(T):
    grad_lstm *= f_gate   # gradient flows through additive cell state
    trace_lstm.append(grad_lstm)
print(f"  Gradient reaching t=1 after {T} steps: {grad_lstm:.3e}")
print(f"  (cell state path multiplies by ~f_t={f_gate}, close to 1, so it survives)\n")

print(f"{'Timestep back':>14} {'RNN gradient':>16} {'LSTM gradient':>16}")
print("-" * 48)
for t in [0, 4, 9, 14, 19, 24, 29]:
    print(f"{t+1:>14} {trace_rnn[t]:>16.3e} {trace_lstm[t]:>16.3e}")

print(f"\nAfter 30 timesteps:")
print(f"  RNN  gradient: {trace_rnn[-1]:.3e}  -> vanished, early inputs forgotten")
print(f"  LSTM gradient: {trace_lstm[-1]:.3e}  -> preserved, long-range memory works")
print(f"\nThis is why LSTMs handle long sequences (long sentences, long audio)")
print(f"while vanilla RNNs struggle to learn long-range dependencies.")
