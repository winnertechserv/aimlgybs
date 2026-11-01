"""
Chapter 17, Example 3: MLFFNN Solving XOR + Architecture Comparison
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
XOR is the canonical example showing why we need hidden layers. A single perceptron (no
hidden layer) gets only 25% accuracy on XOR — it literally cannot draw a line that
separates the two classes. Adding one hidden layer of 4 tanh neurons solves it
instantly: 100% accuracy. We then print out all the learned weights to make the network
transparent — you can trace how input (0,1) flows through the network and comes out as
1. The final section compares three architecture sizes on a real 3-class problem and
shows how adding more parameters doesn't always improve test accuracy (due to
overfitting on a small dataset).

Run it:           python ex3_mlffnn_solving_xor_architecture_comparison.py
Expected output:  expected_output/ex3_mlffnn_solving_xor_architecture_comparison.txt
"""
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

print("=== XOR Problem ===\n")
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0, 1, 1, 0])

# Perceptron: no hidden layers
perc = MLPClassifier(hidden_layer_sizes=(), activation='logistic',
                     max_iter=1000, random_state=42)
perc.fit(X_xor, y_xor)
print(f"Perceptron (0 hidden): acc={accuracy_score(y_xor, perc.predict(X_xor))*100:.0f}%  preds={perc.predict(X_xor)}")

# MLFFNN: 1 hidden layer
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh',
                    max_iter=3000, random_state=1)
mlp.fit(X_xor, y_xor)
print(f"MLFFNN (4 hidden)    : acc={accuracy_score(y_xor, mlp.predict(X_xor))*100:.0f}%  preds={mlp.predict(X_xor)}")

print("\nLearned hidden-layer weights (2 inputs -> 4 neurons):")
for j in range(4):
    w1, w2 = mlp.coefs_[0][0,j], mlp.coefs_[0][1,j]
    b = mlp.intercepts_[0][j]
    print(f"  Neuron {j+1}: x1*{w1:+.3f} + x2*{w2:+.3f} + bias:{b:+.3f}")

print("\n=== Architecture Comparison (3-class task) ===\n")
np.random.seed(42)
X2, y2 = make_classification(n_samples=1000, n_features=10, n_informative=7,
                              n_classes=3, n_clusters_per_class=1, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X2, y2, test_size=0.2, random_state=42)
print(f"{'Architecture':<20} {'Test Acc':>10} {'Total Params':>14}")
print("-" * 48)
for arch in [(10,), (20,10), (50,30,10)]:
    m = MLPClassifier(hidden_layer_sizes=arch, activation='relu',
                      max_iter=600, random_state=42)
    m.fit(Xtr, ytr)
    acc    = accuracy_score(yte, m.predict(Xte))
    params = sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)
    print(f"  {str(arch):<18} {acc*100:>9.1f}% {params:>14}")
