"""
Chapter 30, Example 3: Greedy Search vs Beam Search Decoding
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Decoding strategy is one of the LLM components covered in this chapter. Using a small
language model, greedy search takes the single most likely word at each step and gets
trapped. Beam search instead keeps the B best partial sequences and ends up with a far
higher overall probability. It is a concrete illustration of why beam search is worth
its extra cost.

Run it:           python ex3_greedy_search_vs_beam_search_decoding.py
Expected output:  expected_output/ex3_greedy_search_vs_beam_search_decoding.txt
"""
# Chapter 30 - Example 3: Decoding strategies - Greedy Search vs Beam Search
import numpy as np
np.random.seed(0)

print("=== Decoding Strategies in LLMs: Greedy vs Beam Search ===\n")
print("After the model produces probabilities over the vocabulary at each")
print("step, a DECODING STRATEGY chooses the actual output words.\n")

vocab = ["the", "cat", "sat", "mat", "<end>"]
V = len(vocab)

# A toy language model: P(next word | last word)
P = {
 "<start>": [0.60, 0.25, 0.05, 0.05, 0.05],
 "the":     [0.05, 0.45, 0.10, 0.35, 0.05],
 "cat":     [0.05, 0.05, 0.70, 0.10, 0.10],
 "sat":     [0.50, 0.05, 0.05, 0.30, 0.10],
 "mat":     [0.05, 0.05, 0.05, 0.05, 0.80],
 "<end>":   [0.00, 0.00, 0.00, 0.00, 1.00],
}
print("Toy language model P(next | current):\n")
print(f"  {'current':<10}" + "".join(f"{w:>9}" for w in vocab))
for k, row in P.items():
    print(f"  {k:<10}" + "".join(f"{p:>9.2f}" for p in row))

MAX = 5

# ---------- Greedy search ----------
print("\n--- Greedy Search: always take the single most likely next word ---\n")
seq, cur, logp = [], "<start>", 0.0
for step in range(MAX):
    probs = np.array(P[cur])
    i = int(np.argmax(probs))
    logp += np.log(probs[i])
    cur = vocab[i]
    seq.append(cur)
    print(f"  step {step+1}: pick '{cur}' (p={probs[i]:.2f})   sequence so far: {' '.join(seq)}")
    if cur == "<end>":
        break
greedy_seq, greedy_logp = seq[:], logp
print(f"\n  Greedy result : {' '.join(greedy_seq)}")
print(f"  log-probability: {greedy_logp:.4f}   (probability {np.exp(greedy_logp):.5f})")

# ---------- Beam search ----------
print("\n--- Beam Search: keep the B best partial sequences at every step ---\n")
B = 3
beams = [([], "<start>", 0.0)]
for step in range(MAX):
    cands = []
    for seq, cur, lp in beams:
        if seq and seq[-1] == "<end>":
            cands.append((seq, cur, lp)); continue
        for i, p in enumerate(P[cur]):
            if p <= 0: continue
            cands.append((seq + [vocab[i]], vocab[i], lp + np.log(p)))
    cands.sort(key=lambda t: -t[2])
    beams = cands[:B]
    print(f"  After step {step+1}, top {B} beams:")
    for s, _, lp in beams:
        print(f"     {' '.join(s):<28} logP = {lp:+.4f}")
    if all(s and s[-1] == "<end>" for s, _, _ in beams):
        break

best_seq, _, best_lp = beams[0]
print(f"\n  Beam search result : {' '.join(best_seq)}")
print(f"  log-probability    : {best_lp:.4f}   (probability {np.exp(best_lp):.5f})")

# ---------- Comparison ----------
print("\n=== Comparison ===\n")
print(f"  {'Strategy':<16}{'Sequence':<30}{'logP':>10}")
print("  " + "-" * 56)
print(f"  {'Greedy':<16}{' '.join(greedy_seq):<30}{greedy_logp:>10.4f}")
print(f"  {'Beam (B=3)':<16}{' '.join(best_seq):<30}{best_lp:>10.4f}")
better = "beam search" if best_lp > greedy_logp else "both equal"
print(f"\n  Higher log-probability found by: {better}")
print("\n  Greedy is fast but can be trapped by a locally attractive word.")
print("  Beam search explores B alternatives and usually finds a better")
print("  overall sequence, at B times the computation cost.")
