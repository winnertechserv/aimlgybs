"""
Chapter 2, Example 2: Image Retrieval by Example
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This is query-by-example. We pick eight images as queries and ask the system for the
five most similar images in the collection, using cosine similarity between pixel
vectors. Precision@5 is the share of those five with the same digit as the query. Most
queries score a perfect 1.0. One messy '3' brings back two '8's, which shows that
similar-looking digits are the hard cases. The last line shows query-by-semantics:
because every image is labelled, we can ask for 'sevens' by keyword.

Run it:           python ex2_image_retrieval_by_example.py
Expected output:  expected_output/ex2_image_retrieval_by_example.txt
"""
# Chapter 2 - Example 2: Content-based image retrieval, query-by-example (QBE)
# Give the system one picture; it returns the most similar pictures in the collection.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

digits = load_digits()
X, y = digits.data, digits.target
rng = np.random.default_rng(21)
query_ids = rng.choice(len(X), 8, replace=False)
db_mask = np.ones(len(X), bool); db_mask[query_ids] = False   # queries are not in the database
X_db, y_db = X[db_mask], y[db_mask]

# Cosine similarity: compare the 'direction' of the pixel vectors
index = NearestNeighbors(n_neighbors=5, metric="cosine").fit(normalize(X_db))

precisions = []
for q in query_ids:
    dist, idx = index.kneighbors(normalize(X[q:q + 1]))
    found = y_db[idx[0]]
    p_at_5 = np.mean(found == y[q])
    precisions.append(p_at_5)
    sims = np.round(1 - dist[0], 3)
    print(f"Query digit {y[q]} -> top-5 labels {found.tolist()}  similarity {sims.tolist()}  "
          f"precision@5 = {p_at_5:.1f}")
print(f"\nMean precision@5 over {len(query_ids)} queries: {np.mean(precisions):.2f}")

# Query-by-semantics (QBS) needs labels/annotations instead of an example picture:
want = 7
print(f"\nQBS 'show me sevens': {np.sum(y_db == want)} annotated images match the keyword '{want}'")
