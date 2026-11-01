"""
Chapter 19, Example 1: SGD vs Momentum vs Adam — Optimizer Convergence
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We minimize a 2-D quadratic loss E(w) = w1² + 50·w2² starting at w = [5, 5]. The '50' on
w2 makes a narrow ravine: steep across w2, almost flat along w1. Plain SGD uses one step
size for both weights, and the steep direction sets the limit. The first row shows that
a step of 0.021 makes w2 overshoot further every time and blow up. With a safe step of
0.009, SGD crawls along the flat w1 direction and is still at loss 0.66 after 100 steps.
Momentum keeps the same step size but builds up speed along w1 and finishes far ahead
(0.0009). Adam divides each weight's step by the size of its recent gradients, so the
steep w2 gets a smaller effective step and the flat w1 a larger one, and both move at a
similar pace. It beats SGD clearly (0.024), but on this small, smooth bowl a well-tuned
momentum wins. Adam's real advantage is that it works well across many problems with
little tuning, which is why it is the usual default.

Run it:           python ex1_sgd_vs_momentum_vs_adam_optimizer_convergence.py
Expected output:  expected_output/ex1_sgd_vs_momentum_vs_adam_optimizer_convergence.txt
"""
import numpy as np

print("=== Optimizer Comparison: Minimizing E(w) = w1^2 + 50*w2^2 ===\n")
print("Starting at w=[5.0, 5.0], optimal = [0.0, 0.0]")
print("A narrow ravine: steep in w2 (gradient 100*w2), almost flat in w1 (gradient 2*w1)\n")

loss = lambda w: w[0]**2 + 50*w[1]**2
grad = lambda w: np.array([2*w[0], 100*w[1]])

# SGD: the steep w2 direction caps the step size (above eta = 0.02 it blows up)
def make_sgd(eta=0.009):
    def update(w, g): return w - eta * g
    return update

# Momentum: same step size, but velocity builds up along the flat w1 direction
def make_momentum(eta=0.009, alpha=0.9):
    dw = np.zeros(2)
    def update(w, g):
        nonlocal dw
        dw = -eta * g + alpha * dw
        return w + dw
    return update

# Adam: divides each weight's step by the size of its recent gradients,
# so the steep w2 gets a SMALLER effective step and the flat w1 a LARGER one
def make_adam(eta=0.3, r1=0.9, r2=0.999, eps=1e-8):
    m, v, t = np.zeros(2), np.zeros(2), 0
    def update(w, g):
        nonlocal m, v, t
        t += 1
        m = r1*m + (1-r1)*g
        v = r2*v + (1-r2)*g**2
        mh = m / (1 - r1**t)
        vh = v / (1 - r2**t)
        return w - eta * mh / (np.sqrt(vh) + eps)
    return update

print(f"{'Optimizer':<22} {'Step 10':>10} {'Step 50':>10} {'Step 100':>11}  {'Final w'}")
print("-" * 75)
for name, updater in [('SGD (eta=0.021)',  make_sgd(0.021)),
                      ('SGD (eta=0.009)',  make_sgd()),
                      ('Momentum (a=0.9)', make_momentum()),
                      ('Adam (eta=0.3)',   make_adam())]:
    w = np.array([5.0, 5.0])
    losses = [loss(w)]
    for _ in range(100):
        w = updater(w, grad(w))
        losses.append(loss(w))
    print(f"  {name:<20} {losses[10]:>10.4g} {losses[50]:>10.4g} {losses[100]:>11.4g}  "
          f"[{w[0]:+.4f}, {w[1]:+.4f}]")

print()
print("Observations:")
print("  SGD      : One step size for both weights. Too big and the steep w2 blows up;")
print("             small enough for w2, it crawls along the flat w1 direction")
print("  Momentum : Builds up speed along w1 and finishes far ahead of SGD")
print("  Adam     : Per-weight scaling moves both weights at a similar pace; it beats SGD")
print("             here, but a well-tuned momentum can still win on a smooth toy bowl")
