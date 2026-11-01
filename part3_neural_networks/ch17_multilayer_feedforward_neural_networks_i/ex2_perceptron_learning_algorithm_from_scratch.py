"""
Chapter 17, Example 2: Perceptron Learning Algorithm from Scratch
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We build the perceptron learning rule exactly as described in the overview. The dataset
has two linearly separable classes: class +1 at (1,1), (2,1), (1,2) and class -1 at
(-1,-1), (-2,-1), (-1,-2). We add a bias term by prepending 1 to each input vector. The
algorithm starts with zero weights and updates them only when a misclassification
happens. The training trace shows exactly which example caused an update and what the
new weights are. This converges in just 2 epochs, and the final verification confirms
all 6 examples are classified correctly.

Run it:           python ex2_perceptron_learning_algorithm_from_scratch.py
Expected output:  expected_output/ex2_perceptron_learning_algorithm_from_scratch.txt
"""
import numpy as np

# Augmented inputs [bias=1, x1, x2]
X = np.array([[1,1,1],[1,2,1],[1,1,2],
              [1,-1,-1],[1,-2,-1],[1,-1,-2]])
y = np.array([+1,+1,+1,-1,-1,-1])
w   = np.zeros(3)
eta = 0.5
print("=== Perceptron Learning: Step-by-Step ===\n")
print(f"Initial weights: {w}")
print(f"Learning rate η: {eta}\n")
print(f"{'Epoch':>6} {'Ex':>4} {'True':>6} {'w.x':>8} {'Pred':>6}  New weights")
print("-" * 65)
for epoch in range(1, 15):
    errors = 0
    for i, (xi, yi) in enumerate(zip(X, y)):
        dot  = np.dot(w, xi)
        pred = +1 if dot > 0 else -1
        if pred != yi:
            w += eta * yi * xi
            errors += 1
            print(f"{epoch:>6} {i+1:>4} {yi:>6} {dot:>8.2f} {pred:>6}  "
                  f"[{w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f}]")
    if errors == 0:
        print(f"\nConverged after epoch {epoch}!")
        break

print(f"\nFinal weights: {w}")
print(f"Decision: {w[1]:.2f}*x1 + {w[2]:.2f}*x2 + {w[0]:.2f} = 0\n")
print("Verification:")
for xi, yi in zip(X, y):
    dot  = np.dot(w, xi)
    pred = +1 if dot > 0 else -1
    ok   = "OK" if pred == yi else "WRONG"
    print(f"  ({xi[1]:+d},{xi[2]:+d}) -> w.x={dot:.2f} pred={pred:+d} true={yi:+d} {ok}")
