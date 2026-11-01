"""
Chapter 6, Example 2: Z-Score Normalization & Distance Comparison
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This example shows z-score normalization and — critically — shows WHY normalization
matters for distance-based algorithms. Without normalization, we ask: which salesman is
closest to a query point of (23000 Rs, 6.5 score)? The Euclidean distance is dominated
by the Sales column (values in thousands) making the Score column almost irrelevant.
After z-score normalization, both columns are on equal footing. See how S001, S002, and
S003 get completely different nearest-neighbour rankings before vs after normalization —
S002 becomes the closest, which makes intuitive sense since its sales (23500) and score
(6) are both close to the query (23000, 6.5).

Run it:           python ex2_z_score_normalization_distance_comparison.py
Expected output:  expected_output/ex2_z_score_normalization_distance_comparison.txt
"""
import numpy as np

sales  = np.array([23500, 23500, 22879, 2300, 34678,
                   15687, 18945, 8750, 37489, 73567,
                   52789, 2900, 6570, 21000], dtype=float)
scores = np.array([8, 6, 2, 4, 5, 8, 8, 2, 4, 2, 4, 3, 3, 2], dtype=float)

def zscore(arr):
    return (arr - arr.mean()) / arr.std()

sales_z  = zscore(sales)
scores_z = zscore(scores)

print("=== Z-Score Normalization ===")
print(f"Sales  → mean: {sales.mean():.2f}, std: {sales.std():.2f}")
print(f"Scores → mean: {scores.mean():.2f}, std: {scores.std():.2f}")
print()
print(f"{'ID':<6} {'Sales (Rs)':>12} {'Score':>8} {'Z-Sales':>10} {'Z-Score':>10}")
print("-" * 50)
for i, (s, sc, sn, scn) in enumerate(zip(sales, scores, sales_z, scores_z), 1):
    print(f"S{i:03d}   {s:>12.0f} {sc:>8.0f} {sn:>10.2f} {scn:>10.2f}")
print()
print(f"After normalization: mean={sales_z.mean():.4f}, std={sales_z.std():.4f}")

# Compare distances for query point (23000, 6.5)
query_raw = np.array([23000.0, 6.5])
query_z   = np.array([(23000 - sales.mean())/sales.std(),
                      (6.5   - scores.mean())/scores.std()])

d_raw = np.sqrt((sales - query_raw[0])**2 + (scores - query_raw[1])**2)
d_z   = np.sqrt((sales_z - query_z[0])**2 + (scores_z - query_z[1])**2)

print("\n--- Euclidean Distance to query (23000, 6.5) ---")
print("Before normalization (Sales dominates):")
print(f"  S001: {d_raw[0]:.2f}   S002: {d_raw[1]:.2f}   S003: {d_raw[2]:.2f}")
print("After Z-score normalization (both features contribute equally):")
print(f"  S001: {d_z[0]:.4f}   S002: {d_z[1]:.4f}   S003: {d_z[2]:.4f}")
print("\nS002 is now correctly identified as nearest neighbour.")
