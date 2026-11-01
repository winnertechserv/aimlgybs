"""
Chapter 17, Example 1: Activation Functions Compared
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Before building a neural network, you need to understand what each activation function
does. This example computes and compares Logistic, Tanh, ReLU, and Softmax over a range
of inputs. The key insight is in the derivative table at the bottom: for logistic at
a=±4, the gradient is only 0.0177 — nearly zero. In a deep network with 10 sigmoid
layers, the gradient shrinks by 0.0177^10 ≈ 10^-17 by the time it reaches the first
layer. That's the vanishing gradient problem. ReLU has a derivative of exactly 1 for all
positive inputs — no shrinkage.

Run it:           python ex1_activation_functions_compared.py
Expected output:  expected_output/ex1_activation_functions_compared.txt
"""
import numpy as np

def logistic(a): return 1 / (1 + np.exp(-a))
def tanh(a):     return np.tanh(a)
def relu(a):     return np.maximum(0, a)
def softmax(v):  e = np.exp(v - v.max()); return e / e.sum()

print("=== Activation Functions ===\n")
print(f"{'Input a':>10} {'Logistic':>10} {'Tanh':>10} {'ReLU':>10}")
print(f"{'':>10} {'(0,1)':>10} {'(-1,1)':>10} {'[0,inf)':>10}")
print("-" * 44)
for a in np.linspace(-5, 5, 11):
    print(f"{a:>10.1f} {logistic(a):>10.4f} {tanh(a):>10.4f} {relu(a):>10.4f}")

print()
print("=== Softmax: 4-class probability output ===\n")
raw = np.array([2.0, 1.0, 0.1, -1.5])
probs = softmax(raw)
classes = ['Dog', 'Cat', 'Bird', 'Fish']
print(f"{'Class':<8} {'Raw':>8} {'Prob':>8}")
for cls, r, p in zip(classes, raw, probs):
    print(f"  {cls:<6} {r:>8.1f} {p:>8.4f}  {'X'*int(p*30)}")
print(f"Sum of probs: {probs.sum():.4f}  (always 1.0)")

print()
print("=== Logistic Derivative (used in backprop) ===\n")
print(f"{'Input a':>8} {'f(a)':>10} {'df/da':>10}  Note")
print("-" * 55)
for a in [-4, -2, 0, 2, 4]:
    fa = logistic(a); dfa = fa*(1-fa)
    note = "SATURATED (near-zero gradient!)" if abs(a)>=3 else "Healthy gradient"
    print(f"{a:>8} {fa:>10.4f} {dfa:>10.4f}  {note}")
