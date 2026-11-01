"""
Chapter 1, Example 1: Pattern Recognition — Predicting the Next Number
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This tiny 'learner' looks at a sequence and tries several kinds of rule: multiply by a
constant, add up the previous terms, or a polynomial found by taking differences until
they become constant. It keeps whichever rule fits and uses it to predict the next
number. Look at the last one: 3, 5, 12, 24, 41 has differences 2, 7, 12, 17, which grow
by 5 each time, so the next term is 63.

Run it:           python ex1_pattern_recognition_predicting_the_next_numbe.py
Expected output:  expected_output/ex1_pattern_recognition_predicting_the_next_numbe.txt
"""
# Chapter 1 - Example 1: Pattern recognition - finding the rule behind a sequence
# A tiny "learner" tries a few kinds of rule and keeps the one that fits the data.
import numpy as np

def find_rule(seq):
    s = np.array(seq, dtype=float)
    # Rule 1: constant ratio (geometric, e.g. 1, 2, 4, 8)
    # s[1:] / s[:-1] divides each term by the one before it: [1, 2, 4, 8] -> [2, 2, 2]
    r = s[1:] / s[:-1]
    if np.allclose(r, r[0]):
        return f"multiply by {r[0]:g}", s[-1] * r[0]
    # Rule 2: each term = sum of previous k terms (Fibonacci-like)
    # s[i - k:i] is the k terms just before position i
    for k in (2, 3):
        if all(s[i] == s[i - k:i].sum() for i in range(k, len(s))):
            return f"sum of previous {k} terms", s[-k:].sum()
    # Rule 3: polynomial - take differences until they become constant
    # np.diff subtracts each term from the next one: [1, 4, 9, 16] -> [3, 5, 7]
    diffs, level = [s], 0
    while len(diffs[-1]) > 1 and not np.allclose(diffs[-1], diffs[-1][0]):
        diffs.append(np.diff(diffs[-1])); level += 1
    if len(diffs[-1]) > 1:
        nxt = diffs[-1][0]
        for d in reversed(diffs[:-1]):
            nxt = d[-1] + nxt
        return f"polynomial of degree {level}", nxt
    # Rule 4: primes
    def is_prime(k): return k > 1 and all(k % p for p in range(2, int(k**0.5) + 1))
    if all(is_prime(int(v)) for v in s):
        k = int(s[-1]) + 1
        while not is_prime(k): k += 1
        return "next prime number", k
    return "no rule found", None

sequences = [
    [1, 2, 3, 4, 5],
    [1, 3, 5, 7, 9],
    [2, 3, 5, 7, 11],
    [1, 4, 9, 16, 25],
    [1, 2, 4, 8, 16, 32],
    [1, 1, 2, 3, 5, 8],
    [1, 1, 2, 4, 7, 13],
    [3, 5, 12, 24, 41],
]
for seq in sequences:
    rule, nxt = find_rule(seq)
    print(f"{str(seq):24s} rule: {rule:26s} next -> {nxt:g}")
