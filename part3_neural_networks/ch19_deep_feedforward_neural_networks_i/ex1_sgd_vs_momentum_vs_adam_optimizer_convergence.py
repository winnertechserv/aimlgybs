"""
Chapter 19, Example 1: SGD vs Momentum vs Adam — Optimizer Convergence
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We minimize a simple 2D quadratic loss E(w) = w1² + 10·w2² starting at w=[5,5]. The
'10x' on w2 creates an elongated bowl — a surface where plain SGD struggles because it
takes tiny equal steps in both dimensions. Momentum helps by accumulating velocity in
consistent directions. Adam adapts the learning rate per weight — giving w2 a larger
effective step size since its gradient is large. Tracking the loss at steps 1, 10, 50,
100 shows how quickly each optimizer reaches the minimum [0,0]. Adam gets there fastest
in most real-world scenarios.

Run it:           python ex1_sgd_vs_momentum_vs_adam_optimizer_convergence.py
Expected output:  expected_output/ex1_sgd_vs_momentum_vs_adam_optimizer_convergence.txt
"""
import numpy as np

print("=== Optimizer Comparison: Minimising E(w) = w1^2 + 10*w2^2 ===\n")
print("Starting at w=[5.0, 5.0], optimal = [0.0, 0.0]\n")

loss = lambda w: w[0]**2 + 10*w[1]**2
grad = lambda w: np.array([2*w[0], 20*w[1]])

# SGD
def make_sgd(eta=0.03):
    def update(w, g): return w - eta * g
    return update

# Momentum
def make_momentum(eta=0.03, alpha=0.9):
    dw = np.zeros(2)
    def update(w, g):
        nonlocal dw
        dw = -eta * g + alpha * dw
        return w + dw
    return update

# Adam
def make_adam(eta=0.1, r1=0.9, r2=0.999, eps=1e-8):
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
for name, updater in [('SGD',             make_sgd()),
                       ('Momentum (a=0.9)',make_momentum()),
                       ('Adam',            make_adam())]:
    w = np.array([5.0, 5.0])
    losses = [loss(w)]
    for _ in range(100):
        w = updater(w, grad(w))
        losses.append(loss(w))
    print(f"  {name:<20} {losses[10]:>10.4f} {losses[50]:>10.4f} {losses[100]:>11.6f}  "
          f"[{w[0]:+.4f}, {w[1]:+.4f}]")

print()
print("Observations:")
print("  SGD      : Slow in w2 direction (large gradient dimension)")
print("  Momentum : Accelerates in consistent gradient directions")
print("  Adam     : Per-weight adaptive rates — fastest to optimal")
