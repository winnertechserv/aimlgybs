"""
Chapter 6, Example 1: Min-Max Normalization
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This example walks through Min-Max normalization on a small salesman dataset. We define
a simple function that takes an array, finds its minimum and maximum, then scales every
value proportionally into the [0, 1] range. Notice how the raw sales values (₹2,300 to
₹73,567) and scores (2 to 8) are completely different scales — after normalization, both
live in [0, 1]. The last two rows confirm that the range constraint is satisfied.
Remember: you MUST save the min and max from training and apply them at test time too —
never recompute from test data.

Run it:           python ex1_min_max_normalization.py
Expected output:  expected_output/ex1_min_max_normalization.txt
"""
import numpy as np

# Salesman dataset: Total Sales (Rs) and Score for Sale
sales  = np.array([23500, 23500, 22879,  2300, 34678,
                   15687, 18945,  8750, 37489, 73567,
                   52789,  2900,  6570, 21000], dtype=float)
scores = np.array([8, 6, 2, 4, 5, 8, 8, 2, 4, 2, 4, 3, 3, 2], dtype=float)

def minmax_normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

sales_norm  = minmax_normalize(sales)
scores_norm = minmax_normalize(scores)

print("=== Min-Max Normalization ===")
print(f"Sales  → min: {sales.min():.0f}, max: {sales.max():.0f}")
print(f"Scores → min: {scores.min():.0f}, max: {scores.max():.0f}")
print()
print(f"{'ID':<6} {'Sales (Rs)':>12} {'Score':>8} {'Norm Sales':>12} {'Norm Score':>12}")
print("-" * 55)
for i, (s, sc, sn, scn) in enumerate(zip(sales, scores, sales_norm, scores_norm), 1):
    print(f"S{i:03d}   {s:>12.0f} {sc:>8.0f} {sn:>12.4f} {scn:>12.4f}")
print()
print(f"Norm Sales range : [{sales_norm.min():.4f}, {sales_norm.max():.4f}]")
print(f"Norm Score range : [{scores_norm.min():.4f}, {scores_norm.max():.4f}]")
