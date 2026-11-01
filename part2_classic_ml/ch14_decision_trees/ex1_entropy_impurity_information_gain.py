"""
Chapter 14, Example 1: Entropy Impurity & Information Gain
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This is the most basic calculation in decision tree construction — manual entropy and
information gain computation. We use a small worked example: a node with 10 examples (4
from C1, 4 from C2, 2 from C3) and two competing queries. For each query, we compute
entropy of the parent node and both child nodes, then calculate information gain. Query
1 wins (ΔI = 0.3219 > 0.1568) so it gets selected. The bottom section shows entropy for
special cases — a pure node has entropy 0 (no uncertainty), while equal class
distribution gives maximum entropy.

Run it:           python ex1_entropy_impurity_information_gain.py
Expected output:  expected_output/ex1_entropy_impurity_information_gain.txt
"""
import numpy as np

def entropy(counts):
    total = sum(counts)
    if total == 0: return 0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)

def info_gain(parent, left, right):
    n = sum(parent); nl = sum(left); nr = sum(right)
    return entropy(parent) - (nl/n)*entropy(left) - (nr/n)*entropy(right)

# Node Nd: 4 from C1, 4 from C2, 2 from C3
parent = [4, 4, 2]
print(f"Parent Node: {parent[0]} C1, {parent[1]} C2, {parent[2]} C3")
print(f"Entropy i(Nd) = {entropy(parent):.4f}\n")

# Query 1
left1, right1 = [3, 1, 0], [1, 3, 2]
print(f"Query 1: Left={left1}  Right={right1}")
print(f"  i(NdL)={entropy(left1):.4f}  i(NdR)={entropy(right1):.4f}")
print(f"  Information Gain = {info_gain(parent, left1, right1):.4f}")

# Query 2
left2, right2 = [2, 1, 0], [2, 3, 2]
print(f"\nQuery 2: Left={left2}  Right={right2}")
print(f"  i(NdL)={entropy(left2):.4f}  i(NdR)={entropy(right2):.4f}")
print(f"  Information Gain = {info_gain(parent, left2, right2):.4f}")

ig1 = info_gain(parent, left1, right1)
ig2 = info_gain(parent, left2, right2)
print(f"\nDecision: Query 1 wins ({ig1:.4f} > {ig2:.4f}) -> SELECT Query 1")

print("\n--- Entropy for special cases ---")
print(f"  Pure node [10, 0]  : entropy = {entropy([10,0]):.4f}  (pure, no uncertainty)")
print(f"  50/50 split [5,5]  : entropy = {entropy([5,5]):.4f}  (maximum, 2 classes)")
print(f"  Equal [1,1,1]      : entropy = {entropy([1,1,1]):.4f}  (maximum, 3 classes)")
