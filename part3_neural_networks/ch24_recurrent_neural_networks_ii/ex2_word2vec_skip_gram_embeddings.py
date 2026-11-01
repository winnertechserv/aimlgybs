"""
Chapter 24, Example 2: Word2Vec Skip-Gram Embeddings
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This trains a Skip-Gram model with plain SGD on a small corpus with three semantic
groups (royalty, animals, fruits). After training, the nearest-neighbour words by cosine
similarity cluster by meaning - the model learned semantics purely from context, with no
labels. This is the prediction-based alternative to the count method.

Run it:           python ex2_word2vec_skip_gram_embeddings.py
Expected output:  expected_output/ex2_word2vec_skip_gram_embeddings.txt
"""
# Chapter 24 - Example 2: Word2Vec Skip-Gram intuition with cosine similarity
import numpy as np
np.random.seed(42)

print("=== Word2Vec Skip-Gram: Learning Embeddings by Prediction ===\n")
print("Skip-Gram predicts CONTEXT words from a CENTER word.")
print("Words used in similar contexts end up with similar vectors.\n")

# Tiny corpus with clear semantic groups
sentences = [
    "king queen royal palace throne",
    "king queen royal crown throne",
    "dog cat pet animal furry",
    "dog cat pet animal cute",
    "apple banana fruit sweet juicy",
    "apple banana fruit sweet tasty",
]
tokens = [s.split() for s in sentences]
vocab = sorted({w for s in tokens for w in s})
V = len(vocab)
idx = {w: i for i, w in enumerate(vocab)}
print(f"Vocabulary ({V} words): {vocab}\n")

# Build (center, context) training pairs with window=2
pairs = []
window = 2
for sent in tokens:
    for i, center in enumerate(sent):
        for j in range(max(0,i-window), min(len(sent), i+window+1)):
            if i != j:
                pairs.append((idx[center], idx[sent[j]]))
print(f"Generated {len(pairs)} (center, context) training pairs.\n")

# Skip-gram with simple SGD (embedding dim = 8)
dim = 8
W_in  = np.random.randn(V, dim) * 0.1   # center word embeddings
W_out = np.random.randn(V, dim) * 0.1   # context word embeddings

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

lr = 0.05
for epoch in range(300):
    np.random.shuffle(pairs)
    for c, o in pairs:
        h = W_in[c]                      # center embedding
        scores = W_out @ h               # similarity to all words
        probs = softmax(scores)
        probs[o] -= 1                    # gradient of cross-entropy
        grad_out = np.outer(probs, h)
        grad_in  = W_out.T @ probs
        W_out -= lr * grad_out
        W_in[c] -= lr * grad_in

print("Training complete (300 epochs of SGD).\n")

def cossim(a, b):
    return a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)

def nearest(word, k=3):
    v = W_in[idx[word]]
    sims = [(w, cossim(v, W_in[idx[w]])) for w in vocab if w != word]
    return sorted(sims, key=lambda x: -x[1])[:k]

print("Nearest neighbours by cosine similarity (learned embeddings):")
for w in ['king', 'dog', 'apple']:
    neighbours = nearest(w)
    nbr_str = ", ".join(f"{n}({s:.2f})" for n, s in neighbours)
    print(f"  {w:>6} -> {nbr_str}")

print("\nWords from the same semantic group (royalty / animals / fruits)")
print("cluster together - the model learned meaning from context alone.")
