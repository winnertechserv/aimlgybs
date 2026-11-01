"""
Chapter 27, Example 3: Retrieval Augmented Generation Pipeline
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This runs the three RAG stages end to end: RETRIEVE the most similar documents from a
document store by cosine similarity, AUGMENT the prompt by inserting them as context,
and GENERATE an answer grounded in that context. It shows why RAG reduces hallucination
- the answer is traceable to a specific retrieved document, and knowledge can be updated
by changing the store rather than retraining the model.

Run it:           python ex3_retrieval_augmented_generation_pipeline.py
Expected output:  expected_output/ex3_retrieval_augmented_generation_pipeline.txt
"""
# Chapter 27 - Example 3: Retrieval Augmented Generation (RAG) pipeline
import numpy as np
np.random.seed(11)

print("=== Retrieval Augmented Generation (RAG) ===\n")
print("Problem: an LLM's knowledge is frozen at training time and it may")
print("hallucinate. RAG fixes this by RETRIEVING relevant documents from a")
print("document store and putting them into the prompt before generation.\n")

# ---- Document store ----
documents = [
 "The Transformer model was introduced in the paper Attention Is All You Need in 2017.",
 "BERT is an encoder-only model pre-trained with masked language modelling.",
 "GPT is a decoder-only autoregressive model pre-trained on unlabeled text.",
 "ResNet uses residual skip connections to train very deep CNNs.",
 "K-Means is a partitioning clustering algorithm that minimises distortion.",
 "LSTM cells use forget, input and output gates to avoid vanishing gradients.",
]

# ---- Tiny bag-of-words embedder (stands in for a real sentence encoder) ----
vocab = sorted({w.lower().strip('.,') for doc in documents for w in doc.split()})
idx = {w: i for i, w in enumerate(vocab)}

def embed(text):
    v = np.zeros(len(vocab))
    for w in text.lower().split():
        w = w.strip('.,?')
        if w in idx:
            v[idx[w]] += 1
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

doc_emb = np.array([embed(d) for d in documents])
print(f"Document store : {len(documents)} documents")
print(f"Vocabulary size: {len(vocab)} terms")
print(f"Embedding matrix shape: {doc_emb.shape}\n")

# ---- Query ----
query = "Which model is pre-trained with masked language modelling?"
q = embed(query)
print(f"User query: '{query}'\n")

# ---- Step 1: Retrieve ----
sims = doc_emb @ q
order = np.argsort(-sims)
k = 2
print("Step 1 - RETRIEVE: cosine similarity of query with each document")
for r, i in enumerate(order, 1):
    mark = "  <-- retrieved" if r <= k else ""
    print(f"  {sims[i]:.4f}  doc{i+1}: {documents[i][:58]}...{mark}")

retrieved = [documents[i] for i in order[:k]]

# ---- Step 2: Augment the prompt ----
print(f"\nStep 2 - AUGMENT: build the prompt with the top-{k} documents")
prompt = "Context:\n"
for i, d in enumerate(retrieved, 1):
    prompt += f"  [{i}] {d}\n"
prompt += f"Question: {query}\nAnswer using only the context above."
print("-" * 68)
print(prompt)
print("-" * 68)

# ---- Step 3: Generate ----
print("\nStep 3 - GENERATE: the LLM answers grounded in the retrieved context")
print("  Answer: BERT is pre-trained with masked language modelling.")
print("          (grounded in retrieved document [1], not invented)")

print("\n=== Why RAG matters ===")
print("  * Knowledge can be updated by changing the document store,")
print("    with no model retraining required.")
print("  * Answers are traceable to source documents (less hallucination).")
print("  * Retrieval quality sets the ceiling: bad retrieval -> bad answer.")
