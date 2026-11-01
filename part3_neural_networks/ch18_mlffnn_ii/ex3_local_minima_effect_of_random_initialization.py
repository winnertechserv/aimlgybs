"""
Chapter 18, Example 3: Local Minima — Effect of Random Initialization
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
One of the most practically important issues in neural network training is that
different random initializations lead to different final models — because gradient
descent can get stuck in different local minima. This example trains the same MLFFNN 10
times on the same data with 10 different random seeds. The resulting test accuracies
vary noticeably (the range between best and worst is printed). This is why frameworks
like PyTorch let you set random seeds for reproducibility, and why practitioners run
multiple experiments and pick the best model. Pick it the right way: the data is split
three ways, the best seed is chosen on the validation set, and the test set is used only
once, on the chosen model. Notice that the chosen model scores 96% on validation but 91%
on test. Picking the best of ten runs flatters the validation score a little, which is
exactly why the final number must come from data that played no part in the choice.

Run it:           python ex3_local_minima_effect_of_random_initialization.py
Expected output:  expected_output/ex3_local_minima_effect_of_random_initialization.txt
"""
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)   # some seeds need more than 500 epochs

X, y = make_moons(n_samples=500, noise=0.25, random_state=0)
# Three-way split: train (60%) to fit, validation (20%) to choose, test (20%) to report once
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print("=== Local Minima: Effect of Random Initialization ===\n")
print("Same network, same data, different random seeds -> different results")
print(f"Architecture: (3,) | Activation: tanh | Optimizer: SGD")
print(f"Split: {len(X_train)} train / {len(X_val)} validation / {len(X_test)} test\n")
print(f"{'Run':>5} {'Seed':>6} {'Train Acc':>11} {'Val Acc':>10}  {'Status'}")
print("-" * 52)
models, val_accs = [], []
for run, seed in enumerate(range(1, 11), 1):
    mlp = MLPClassifier(hidden_layer_sizes=(3,), activation='tanh',
                        solver='sgd', max_iter=500, random_state=seed,
                        learning_rate_init=0.05)
    mlp.fit(X_train, y_train)
    tr = accuracy_score(y_train, mlp.predict(X_train))
    va = accuracy_score(y_val,   mlp.predict(X_val))
    status = "BEST so far" if not val_accs or va > max(val_accs) else ""
    models.append(mlp); val_accs.append(va)
    print(f"{run:>5} {seed:>6} {tr*100:>10.1f}% {va*100:>9.1f}%  {status}")

best = int(np.argmax(val_accs))
print(f"\nBest  validation accuracy: {max(val_accs)*100:.1f}%  (seed {best + 1})")
print(f"Worst validation accuracy: {min(val_accs)*100:.1f}%")
print(f"Range                    : {(max(val_accs)-min(val_accs))*100:.1f}%")

# The test set is used exactly once, on the model chosen by validation accuracy
test_acc = accuracy_score(y_test, models[best].predict(X_test))
print(f"\nTest accuracy of the chosen model (reported once): {test_acc*100:.1f}%")
print("\nConclusion: run several random initializations, pick the best on VALIDATION data,")
print("and touch the test set only once at the end.")
