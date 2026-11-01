"""
Chapter 5, Example 1: Profiling a Dataset
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This is the first thing you do with any new dataset. The program builds a small
passenger table, splits the columns into numbers and categories, then measures the
centre and spread of 'age'. Look at the skewness lines. Age is roughly symmetric, so its
mean and median are close. Fare is strongly right-skewed: its mean (31) is well above
its median (21) because a few people paid very high fares.

Run it:           python ex1_profiling_a_dataset.py
Expected output:  expected_output/ex1_profiling_a_dataset.txt
"""
# Chapter 5 - Example 1: Profiling a dataset (data types, centre, spread, shape)
import numpy as np
import pandas as pd
from scipy.stats import skew

# A small passenger dataset, built in code so it runs anywhere (no download)
rng = np.random.default_rng(42)
n = 200
df = pd.DataFrame({
    "age":      np.round(rng.normal(30, 12, n).clip(1, 75)),
    "fare":     np.round(rng.lognormal(3.0, 0.9, n), 2),   # long right tail
    "pclass":   rng.choice([1, 2, 3], n, p=[0.25, 0.25, 0.5]),
    "sex":      rng.choice(["male", "female"], n, p=[0.6, 0.4]),
    "embarked": rng.choice(["S", "C", "Q"], n, p=[0.7, 0.2, 0.1]),
})

# 1. What kind of data is each column?
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]
print("Numerical columns  :", num_cols)
print("Categorical columns:", cat_cols)

# 2. Central tendency and dispersion for 'age'
age = df["age"]
print("\n--- Age: centre and spread ---")
print(f"Mean   : {age.mean():.2f}")
print(f"Median : {age.median():.2f}")
print(f"Mode   : {age.mode()[0]:.0f}")
print(f"Var    : {age.var():.2f}")
print(f"Std    : {age.std():.2f}")
print(f"Range  : {age.max() - age.min():.0f}")

# 3. Five-point summary (min, Q1, median, Q3, max)
q = age.quantile([0, 0.25, 0.5, 0.75, 1.0])
print("\n--- Five-point summary of age ---")
for name, v in zip(["Min", "Q1", "Median", "Q3", "Max"], q):
    print(f"{name:7s}: {v:.1f}")

# 4. Skewness tells us which way the tail points
for col in ["age", "fare"]:
    s = skew(df[col])
    shape = "right-skewed" if s > 0.5 else "left-skewed" if s < -0.5 else "roughly symmetric"
    print(f"\nSkewness of {col}: {s:.2f} -> {shape}")
    print(f"  mean = {df[col].mean():.2f}, median = {df[col].median():.2f}")

# 5. Categorical columns: counts are the summary
print("\nPassengers per class:")
print(df["pclass"].value_counts().sort_index().to_string())
