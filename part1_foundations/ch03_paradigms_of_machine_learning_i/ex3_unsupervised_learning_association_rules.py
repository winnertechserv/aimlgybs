"""
Chapter 3, Example 3: Unsupervised Learning — Association Rules
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Ten shopping baskets, no labels. The program counts how often items appear together and
turns each pair into rules with support, confidence and lift. 'banana → cereal' has
confidence 1.00 and lift 2.50: everyone who bought bananas also bought cereal. 'milk →
bread' has a lift below 1, so those two items are not really linked, even though both
are popular.

Run it:           python ex3_unsupervised_learning_association_rules.py
Expected output:  expected_output/ex3_unsupervised_learning_association_rules.txt
"""
# Chapter 3 - Example 3: Unsupervised learning - association rules (market-basket analysis)
# Find products that are bought together, using support, confidence and lift.
from itertools import combinations

baskets = [
    {"bread", "milk"}, {"bread", "butter", "milk"}, {"bread", "butter"},
    {"milk", "cereal"}, {"bread", "butter", "jam"}, {"milk", "cereal", "banana"},
    {"bread", "butter", "milk"}, {"cereal", "banana"}, {"bread", "jam"},
    {"milk", "cereal", "bread"},
]
N = len(baskets)
items = sorted(set().union(*baskets))

def support(itemset):
    # For Python sets, a <= b means 'a is a subset of b' (every item of a is in b)
    return sum(itemset <= b for b in baskets) / N   # share of baskets containing ALL items

print("Item supports:", {i: support({i}) for i in items})

rules = []
for a, b in combinations(items, 2):
    s_ab = support({a, b})
    if s_ab < 0.2:                       # ignore rare pairs (minimum support = 20%)
        continue
    for x, z in ((a, b), (b, a)):
        conf = s_ab / support({x})       # P(z in basket | x in basket)
        lift = conf / support({z})       # >1 means 'more often together than by chance'
        rules.append((lift, conf, s_ab, x, z))

print(f"\nRules with support >= 20% (sorted by lift):")
print("  rule                    support  confidence  lift")
for lift, conf, s, x, z in sorted(rules, reverse=True):
    print(f"  {x:>7s} -> {z:<12s}   {s:.2f}      {conf:.2f}     {lift:.2f}")
