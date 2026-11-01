"""
Chapter 4, Example 2: Active Learning vs Random Labeling
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Both strategies start with 20 labeled images and add 10 labels per round. The random
strategy labels any 10 images. The active strategy asks the 'expert' to label the 10
images where the model's top two guesses are closest, which is uncertainty sampling.
Averaged over five runs, active learning pulls ahead from the first round. With 100
labels it scores 92.7% against 88.6%, so the same labeling budget buys a better model.

Run it:           python ex2_active_learning_vs_random_labeling.py
Expected output:  expected_output/ex2_active_learning_vs_random_labeling.txt
"""
# Chapter 4 - Example 2: Active learning - ask the expert to label the MOST useful images
# Uncertainty sampling picks the images the model is least sure about.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = load_digits(return_X_y=True)
X = X / 16.0
X_pool, X_te, y_pool, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

def run(strategy, seed=0, start=20, batch=10, rounds=8):
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(len(X_pool), start, replace=False))
    scores = []
    for _ in range(rounds + 1):
        m = LogisticRegression(max_iter=2000).fit(X_pool[labeled], y_pool[labeled])
        scores.append(m.score(X_te, y_te))
        rest = np.setdiff1d(np.arange(len(X_pool)), labeled)
        if strategy == "uncertainty":
            p = np.sort(m.predict_proba(X_pool[rest]), axis=1)
            margin = p[:, -1] - p[:, -2]              # small gap = model is unsure
            pick = rest[np.argsort(margin)[:batch]]
        else:
            pick = rng.choice(rest, batch, replace=False)
        labeled += list(pick)                        # the 'expert' supplies these labels
    return np.array(scores)

active = np.mean([run("uncertainty", s) for s in range(5)], axis=0)
random_ = np.mean([run("random", s) for s in range(5)], axis=0)
print("Labels used   Random sampling   Active (uncertainty)")
for i, (r, a) in enumerate(zip(random_, active)):
    print(f"   {20 + 10 * i:3d}           {r:.3f}             {a:.3f}")
print(f"\nWith 100 labels, active learning is {100 * (active[-1] - random_[-1]):.1f} "
      "percentage points more accurate (average of 5 runs).")
