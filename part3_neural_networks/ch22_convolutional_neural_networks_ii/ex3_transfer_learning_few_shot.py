"""
Chapter 22, Example 3: Transfer Learning (Few-Shot)
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This shows transfer learning. We pretrain a deep model on a large set of digit classes,
then tackle two brand-new classes with only 10 labeled images. Reusing the pretrained
features and training a tiny classifier on top beats training a fresh model from
scratch. This mirrors how a VGGNet pretrained on ImageNet is reused for medical images
with little labeled data.

Run it:           python ex3_transfer_learning_few_shot.py
Expected output:  expected_output/ex3_transfer_learning_few_shot.txt
"""
# Chapter 22 - Example 3: Transfer learning simulation (few-shot target domain)
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(0)

print("=== Transfer Learning: Reusing a Pretrained Representation ===\n")

digits = load_digits()
X, y = digits.data, digits.target

# Source domain: a LARGE labeled dataset (8 of the 10 digit classes)
src_classes = [0,1,2,3,4,5,6,7]
src = np.isin(y, src_classes)
Xs, ys = X[src], y[src]
print(f"Source domain : digits {src_classes}")
print(f"                {len(Xs)} labeled images -> pretrain a deep CNN-like model")
pretrained = MLPClassifier(hidden_layer_sizes=(120, 64), max_iter=600, random_state=0)
pretrained.fit(Xs, ys)
print(f"                pretrained source accuracy: {pretrained.score(Xs, ys)*100:.1f}%\n")

# Target domain: NEW classes (8, 9) with only a handful of labels each
tgt = np.isin(y, [8, 9])
Xt, yt = X[tgt], y[tgt]
# Only 5 examples per class = 10 total labeled target images (few-shot)
Xt_train, Xt_test, yt_train, yt_test = train_test_split(
    Xt, yt, train_size=10, random_state=0, stratify=yt)
print(f"Target domain : NEW digits 8 and 9")
print(f"                only {len(Xt_train)} labeled images (5 per class - few-shot!)\n")

# Pretrained feature extractor (hidden activations)
def extract_features(model, Xin):
    a = Xin
    for W, b in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        a = np.maximum(0, a @ W + b)
    return a

ft_tr = extract_features(pretrained, Xt_train)
ft_te = extract_features(pretrained, Xt_test)

# Approach 1: transfer - simple classifier on pretrained features
transfer = LogisticRegression(max_iter=2000, random_state=0)
transfer.fit(ft_tr, yt_train)
acc_transfer = accuracy_score(yt_test, transfer.predict(ft_te))

# Approach 2: from scratch on raw pixels, same 10 examples
scratch = MLPClassifier(hidden_layer_sizes=(120, 64), max_iter=600, random_state=0)
scratch.fit(Xt_train, yt_train)
acc_scratch = accuracy_score(yt_test, scratch.predict(Xt_test))

print(f"Test set: {len(yt_test)} unseen images of digits 8 and 9\n")
print("Results:")
print(f"  Train from scratch (raw pixels, 10 imgs) : {acc_scratch*100:.1f}%")
print(f"  Transfer learning (pretrained features)  : {acc_transfer*100:.1f}%")
print(f"\nImprovement from transfer learning: {(acc_transfer-acc_scratch)*100:+.1f} pp")
print("\nWith only 10 labeled images, training a deep model from scratch fails to")
print("generalize. The pretrained model already learned reusable stroke/edge")
print("features, so a tiny classifier on top of them generalizes far better.")
print("This is exactly how a VGGNet pretrained on ImageNet is reused for")
print("medical images with only a few labeled examples.")
