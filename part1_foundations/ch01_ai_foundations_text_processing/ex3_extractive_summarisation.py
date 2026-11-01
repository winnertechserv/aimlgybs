"""
Chapter 1, Example 3: Extractive Summarisation
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This is the simplest extractive summariser. It counts how often each meaningful word
appears (ignoring filler words such as 'the'), scores each sentence by the average
importance of its words, and keeps the top three in their original order. Notice that
the off-topic sentence about the weather gets a low score and is left out.

Run it:           python ex3_extractive_summarisation.py
Expected output:  expected_output/ex3_extractive_summarisation.txt
"""
# Chapter 1 - Example 3: Extractive text summarisation
# Score each sentence by how many important (frequent, non-filler) words it contains,
# then keep the top sentences in their original order.
import re
from collections import Counter

document = """Machine translation converts text from one language into another.
It is hard because languages differ in word order and grammar.
German, for example, usually puts the verb in a different place from Japanese.
Modern translation systems learn from millions of translated sentence pairs.
The weather was pleasant on the day the first system was demonstrated.
These systems use neural networks that read the whole sentence before translating it.
Good translation must keep both the meaning and the style of the original text."""

STOP = set("""a an the is it in of from to and or on was were be that this for
by with must both before its one into usually""".split())

sentences = [s.strip() for s in document.split("\n") if s.strip()]
words = [w for w in re.findall(r"[a-z]+", document.lower()) if w not in STOP]
freq = Counter(words)
top = max(freq.values())
weight = {w: c / top for w, c in freq.items()}          # normalise to 0..1
print("Most important words:", [w for w, _ in freq.most_common(6)])

scores = []
for i, s in enumerate(sentences):
    toks = [w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP]
    scores.append((sum(weight[w] for w in toks) / max(len(toks), 1), i))

print("\nSentence scores:")
for sc, i in scores:
    print(f"  {sc:.2f}  {sentences[i][:62]}")

k = 3
chosen = sorted(i for _, i in sorted(scores, reverse=True)[:k])
print(f"\nSummary ({k} of {len(sentences)} sentences):")
for i in chosen:
    print(" -", sentences[i])
