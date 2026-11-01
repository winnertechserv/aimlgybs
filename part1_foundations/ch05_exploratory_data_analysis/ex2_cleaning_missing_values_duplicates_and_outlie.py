"""
Chapter 5, Example 2: Cleaning — Missing Values, Duplicates and Outliers
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We plant three kinds of problem in a clean dataset and then fix them. Compare the five
ways of filling the four missing weights. Mean and median give every row the same value.
KNN imputation uses each person's height, so a shorter person gets a lower weight, which
is the most sensible result. Both the Z-score and IQR tests catch the 250 kg typing
error. The IQR test also flags one genuinely light person (43.4 kg), a reminder that not
every outlier is a mistake. The last lines show how removing, capping or log-
transforming the outliers changes the numbers.

Run it:           python ex2_cleaning_missing_values_duplicates_and_outlie.py
Expected output:  expected_output/ex2_cleaning_missing_values_duplicates_and_outlie.txt
"""
# Chapter 5 - Example 2: Cleaning data - missing values, duplicates and outliers
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.impute import KNNImputer

rng = np.random.default_rng(7)
n = 100
df = pd.DataFrame({"height": rng.normal(170, 8, n).round(1)})
# taller people tend to be heavier, plus some random variation
df["weight"] = (0.9 * (df["height"] - 170) + 70 + rng.normal(0, 4, n)).round(1)

# Plant some problems on purpose
df.loc[[3, 15, 40, 77], "weight"] = np.nan   # missing values
df.loc[60, "weight"] = 250.0                 # a typing error -> outlier
df = pd.concat([df, df.iloc[[5, 9]]], ignore_index=True)  # 2 duplicate rows

print("Rows:", len(df), "| missing weights:", df["weight"].isna().sum(),
      "| duplicates:", df.duplicated().sum())

# 1. Remove duplicates (keep the first copy)
df = df.drop_duplicates(keep="first").reset_index(drop=True)
print("After drop_duplicates:", len(df), "rows")

# 2. Compare ways of filling the gaps
true_missing = df["weight"].isna()
filled = {
    "Mean":          df["weight"].fillna(df["weight"].mean()),
    "Median":        df["weight"].fillna(df["weight"].median()),
    "Forward fill":  df["weight"].ffill(),
    "Interpolation": df["weight"].interpolate(),
    "KNN (k=3)":     pd.Series(KNNImputer(n_neighbors=3).fit_transform(df)[:, 1]),
}
print("\nFilled values for the 4 missing rows:")
for name, s in filled.items():
    print(f"  {name:14s}", np.round(s[true_missing].values, 1))
print("  Heights of those rows:", df.loc[true_missing, "height"].values)

df["weight"] = filled["KNN (k=3)"]   # KNN uses height, so it is the most sensible

# 3. Outliers by Z-score (|z| > 3)
z = zscore(df["weight"])
print("\nZ-score outliers:", df.loc[np.abs(z) > 3, "weight"].tolist())

# 4. Outliers by IQR (beyond 1.5 x IQR from the quartiles)
q1, q3 = df["weight"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"IQR fences: [{lo:.1f}, {hi:.1f}]")
print("IQR outliers:", df.loc[(df["weight"] < lo) | (df["weight"] > hi), "weight"].tolist())

# 5. Three ways to deal with the outlier
removed = df[(df["weight"] >= lo) & (df["weight"] <= hi)]
capped = df["weight"].clip(df["weight"].quantile(0.05), df["weight"].quantile(0.95))
logged = np.log1p(df["weight"])
print(f"\nMean weight  raw: {df['weight'].mean():.2f}  removed: {removed['weight'].mean():.2f}"
      f"  winsorized: {capped.mean():.2f}")
print(f"Max after log transform: {logged.max():.2f} (raw max {df['weight'].max():.0f})")
