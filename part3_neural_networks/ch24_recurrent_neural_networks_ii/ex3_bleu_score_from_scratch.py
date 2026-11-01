"""
Chapter 24, Example 3: BLEU Score from Scratch
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This builds the BLEU metric used to score generated text against a reference: n-gram
precision combined with a brevity penalty. A perfect match scores 1.0; a too-short
output is penalized heavily; a reworded sentence scores partially. The summary contrasts
BLEU with METEOR, ROUGE, and CIDEr.

Run it:           python ex3_bleu_score_from_scratch.py
Expected output:  expected_output/ex3_bleu_score_from_scratch.txt
"""
# Chapter 24 - Example 3: BLEU score from scratch (text-generation evaluation)
import numpy as np
from collections import Counter

print("=== BLEU Score from Scratch (Machine Translation / Captioning metric) ===\n")
print("BLEU = precision-based n-gram matching between generated & reference text,")
print("multiplied by a brevity penalty for too-short outputs.\n")

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def modified_precision(cand, ref, n):
    cand_ng = Counter(ngrams(cand, n))
    ref_ng  = Counter(ngrams(ref, n))
    if not cand_ng:
        return 0.0
    clipped = sum(min(cnt, ref_ng[ng]) for ng, cnt in cand_ng.items())
    return clipped / sum(cand_ng.values())

def bleu(candidate, reference, max_n=2):
    cand, ref = candidate.split(), reference.split()
    print(f"  Candidate: '{candidate}'")
    print(f"  Reference: '{reference}'")
    precisions = []
    for n in range(1, max_n+1):
        p = modified_precision(cand, ref, n)
        precisions.append(p)
        print(f"    {n}-gram precision: {p:.4f}")
    geo_mean = (np.exp(np.mean([np.log(p) for p in precisions]))
                if min(precisions) > 0 else 0.0)
    c, r = len(cand), len(ref)
    bp = 1.0 if c > r else (np.exp(1 - r/c) if c > 0 else 0.0)
    score = bp * geo_mean
    print(f"    geo-mean={geo_mean:.4f}  brevity-penalty={bp:.4f} (c={c}, r={r})")
    print(f"    => BLEU-{max_n} = {score:.4f}\n")
    return score

print("--- Example 1: Perfect match (BLEU-4) ---")
bleu("the cat is on the mat", "the cat is on the mat", max_n=4)

print("--- Example 2: Partially correct (BLEU-2) ---")
bleu("the cat sat on mat", "the cat is on the mat", max_n=2)

print("--- Example 3: Too short - brevity penalty (BLEU-2) ---")
bleu("the cat", "the cat is on the mat", max_n=2)

print("--- Example 4: Good but reworded (BLEU-2) ---")
bleu("the cat is on the mat", "a cat sits on the mat", max_n=2)

print("="*58)
print("BLEU ranges 0..1. Perfect match = 1.0. Missing higher-order")
print("n-grams or too-short output sharply lowers the score.")
print("BLEU = precision-focused; METEOR adds recall + synonyms;")
print("ROUGE = recall-focused (summaries); CIDEr weights rare n-grams.")
