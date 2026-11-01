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
multiple experiments and pick the best model.

Run it:           python ex3_local_minima_effect_of_random_initialization.py
Expected output:  expected_output/ex3_local_minima_effect_of_random_initialization.txt
"""
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = make_moons(n_samples=500, noise=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("=== Local Minima: Effect of Random Initialization ===\n")
print("Same network, same data, different random seeds -> different results")
print(f"Architecture: (3,) | Activation: tanh | Optimizer: SGD\n")
print(f"{'Run':>5} {'Seed':>6} {'Train Acc':>11} {'Test Acc':>11}  {'Status'}")
print("-" * 55)
accs = []
for run, seed in enumerate(range(1, 11), 1):
    mlp = MLPClassifier(hidden_layer_sizes=(3,), activation='tanh',
                        solver='sgd', max_iter=500, random_state=seed,
                        learning_rate_init=0.05)
    mlp.fit(X_train, y_train)
    tr  = accuracy_score(y_train, mlp.predict(X_train))
    te  = accuracy_score(y_test,  mlp.predict(X_test))
    accs.append(te)
    status = "BEST so far" if te == max(accs) else ""
    print(f"{run:>5} {seed:>6} {tr*100:>10.1f}% {te*100:>10.1f}%  {status}")

print(f"\nBest  test accuracy: {max(accs)*100:.1f}%")
print(f"Worst test accuracy: {min(accs)*100:.1f}%")
print(f"Range              : {(max(accs)-min(accs))*100:.1f}%")
print(f"\nConclusion: Always run multiple random initializations and pick the best model.")
