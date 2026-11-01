"""
Chapter 18, Example 2: Training Modes Compared + Early Stopping
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Sklearn's MLPClassifier lets you control the batch size, effectively simulating Pattern
Mode (batch=1), Mini-Batch Mode (batch=32, 64, 128), and Batch Mode (batch=full
dataset). The results confirm the theory: Mini-Batch mode converges in far fewer epochs
than full-batch, and Adam with mini-batch=32 is the fastest of all (only 45 epochs to
converge). The early stopping demo shows that with n_iter_no_change=10 (stop if no
improvement for 10 consecutive epochs), training halts at epoch 55 instead of running
all 300 — a 5x speedup with comparable accuracy.

Run it:           python ex2_training_modes_compared_early_stopping.py
Expected output:  expected_output/ex2_training_modes_compared_early_stopping.txt
"""
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X, y = make_classification(n_samples=2000, n_features=15, n_informative=10,
                            n_classes=2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr2 = scaler.fit_transform(X_tr2)
X_va  = scaler.transform(X_va)
X_te  = scaler.transform(X_te)

print("=== Training Mode Comparison ===\n")
print(f"{'Mode':<26} {'Train':>7} {'Test':>7} {'Epochs':>8}")
print("-" * 52)
configs = [
    ('Batch (full dataset)', 'sgd', len(X_tr2)),
    ('Mini-Batch size=32',   'sgd', 32),
    ('Mini-Batch size=64',   'sgd', 64),
    ('Mini-Batch size=128',  'sgd', 128),
    ('Adam (batch=32)',      'adam', 32),
]
for name, solver, bs in configs:
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu',
                        solver=solver, batch_size=bs,
                        max_iter=300, random_state=42, learning_rate_init=0.01)
    mlp.fit(X_tr2, y_tr2)
    tr = accuracy_score(y_tr2, mlp.predict(X_tr2))
    te = accuracy_score(y_te,  mlp.predict(X_te))
    print(f"  {name:<24} {tr*100:>6.1f}% {te*100:>6.1f}% {mlp.n_iter_:>8}")

print("\n=== Early Stopping Demo ===\n")
for es in [False, True]:
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu',
                        solver='adam', max_iter=300, random_state=42,
                        early_stopping=es, validation_fraction=0.15, n_iter_no_change=10)
    mlp.fit(X_tr2, y_tr2)
    te = accuracy_score(y_te, mlp.predict(X_te))
    label = "Early Stopping ON " if es else "Early Stopping OFF"
    print(f"  {label}: epochs={mlp.n_iter_:>4} | Test={te*100:.1f}%")
print("\nEarly stopping: fewer epochs, same performance, less compute wasted.")
