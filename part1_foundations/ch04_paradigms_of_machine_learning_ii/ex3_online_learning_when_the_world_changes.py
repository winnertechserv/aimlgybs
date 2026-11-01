"""
Chapter 4, Example 3: Online Learning When the World Changes
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This simulates a data stream where the rule suddenly flips at batch 10, which is concept
drift. The static model was trained once at the start and never updated, so after the
drift it falls to coin-flip accuracy (about 50%). The online model is updated with
partial_fit after every batch. It stumbles for one batch, then recovers to about 97%.
Note the order inside the loop: each batch is first used to test, then to learn, exactly
as in a live system.

Run it:           python ex3_online_learning_when_the_world_changes.py
Expected output:  expected_output/ex3_online_learning_when_the_world_changes.txt
"""
# Chapter 4 - Example 3: Online learning on a data stream that changes (concept drift)
# A model trained once goes stale; a model updated batch-by-batch keeps up.
import numpy as np
from sklearn.linear_model import SGDClassifier

rng = np.random.default_rng(0)

def make_batch(t, n=200):
    """Two features. Before t=10 the rule is x1 + x2 > 0; after t=10 the rule flips."""
    X = rng.normal(0, 1, (n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    if t >= 10:
        y = (X[:, 0] - X[:, 1] > 0).astype(int)   # the world has changed (drift)
    return X, y

static = SGDClassifier(loss="log_loss", random_state=0)
online = SGDClassifier(loss="log_loss", random_state=0)
X0, y0 = make_batch(0)
static.fit(X0, y0)                                  # trained once, then frozen
online.partial_fit(X0, y0, classes=[0, 1])

print("Batch   Static model   Online model")
for t in range(1, 20):
    X, y = make_batch(t)
    s_acc, o_acc = static.score(X, y), online.score(X, y)  # test on new data first...
    online.partial_fit(X, y)                               # ...then learn from it
    if t % 2 == 1 or t in (10, 11):
        flag = "  <- drift happens here" if t == 10 else ""
        print(f"  {t:2d}      {s_acc:.2f}          {o_acc:.2f}{flag}")
