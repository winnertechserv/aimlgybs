"""
Chapter 24, Example 1: Count-Based Word Embedding (Co-occurrence to SVD)
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This builds word vectors the count-based way: build a co-occurrence matrix from a tiny
corpus, convert it to Positive PMI, then apply SVD to get compact 2-D vectors. The
cosine similarities confirm that words used in similar contexts ('cat' and 'dog') end up
with similar vectors. This captures global co-occurrence statistics.

Run it:           python ex1_count_based_word_embedding_co_occurrence_to_s.py
Expected output:  expected_output/ex1_count_based_word_embedding_co_occurrence_to_s.txt
"""
# Chapter 24 - Example 1: Count-based word embedding (co-occurrence -> PMI -> SVD)
import numpy as np
np.random.seed(0)

print("=== Count-Based Word Embedding: Co-occurrence -> PPMI -> SVD ===\n")

# A tiny corpus
corpus = ("the cat sat on the mat . the dog sat on the rug . "
          "the cat ran . the dog ran . the cat and the dog play").split()
print("Corpus:", " ".join(corpus), "\n")

# Build vocabulary
vocab = sorted(set(corpus))
V = len(vocab)
idx = {w: i for i, w in enumerate(vocab)}
print(f"Vocabulary ({V} words): {vocab}\n")

# Co-occurrence matrix with context window = 1 (neighbors)
C = np.zeros((V, V))
window = 1
for i, w in enumerate(corpus):
    for j in range(max(0, i-window), min(len(corpus), i+window+1)):
        if i != j:
            C[idx[w], idx[corpus[j]]] += 1

print("Co-occurrence counts computed (window=1).")

# Positive Pointwise Mutual Information (PPMI)
total = C.sum()
row_sum = C.sum(axis=1, keepdims=True)
col_sum = C.sum(axis=0, keepdims=True)
with np.errstate(divide='ignore', invalid='ignore'):
    pmi = np.log2((C * total) / (row_sum @ col_sum))
    pmi[np.isnan(pmi)] = 0
    pmi[np.isinf(pmi)] = 0
ppmi = np.maximum(pmi, 0)   # positive PMI only
print("PPMI matrix computed (negative values clipped to 0).\n")

# SVD to get low-dimensional embeddings
U, S, Vt = np.linalg.svd(ppmi)
r = 2   # embedding dimension
embeddings = U[:, :r] * S[:r]   # W = U_hat * S_hat
print(f"SVD applied. Keeping top r={r} dimensions.")
print(f"PPMI matrix size: {ppmi.shape} -> Word embeddings: {embeddings.shape}\n")

print("2-D word embeddings (first 6 words):")
for w in vocab[:6]:
    vec = embeddings[idx[w]]
    print(f"  {w:>6}: [{vec[0]:+.3f}, {vec[1]:+.3f}]")

# Cosine similarity between related words
def cossim(a, b):
    return a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)

print("\nCosine similarity between word pairs:")
for w1, w2 in [('cat','dog'), ('sat','ran'), ('cat','mat')]:
    s = cossim(embeddings[idx[w1]], embeddings[idx[w2]])
    print(f"  sim({w1}, {w2}) = {s:+.3f}")
print("\n'cat' and 'dog' appear in similar contexts -> similar vectors.")
print("Count-based methods capture GLOBAL co-occurrence statistics.")
