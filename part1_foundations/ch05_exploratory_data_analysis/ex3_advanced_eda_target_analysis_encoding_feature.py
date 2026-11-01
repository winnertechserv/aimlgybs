"""
Chapter 5, Example 3: Advanced EDA — Target Analysis, Encoding, Features and Leakage
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This mirrors the classic Titanic case study, using made-up data. Grouping by class and
sex shows who was more likely to survive. We then encode categories, engineer family
size, age groups and a log fare, and rank every feature by its correlation with
survival. One column, 'lifeboat_assigned', has a correlation of 0.97. It looks like a
perfect predictor, but it is only known after the event. The program flags it as data
leakage.

Run it:           python ex3_advanced_eda_target_analysis_encoding_feature.py
Expected output:  expected_output/ex3_advanced_eda_target_analysis_encoding_feature.txt
"""
# Chapter 5 - Example 3: Advanced EDA - correlation, target-wise analysis,
# encoding, feature engineering and a data-leakage check
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
n = 600
pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.25, 0.5])
sex = rng.choice(["male", "female"], n, p=[0.6, 0.4])
age = rng.normal(30, 12, n).clip(1, 75).round()
sibsp = rng.poisson(0.5, n)
parch = rng.poisson(0.4, n)
fare = np.round(np.where(pclass == 1, 80, np.where(pclass == 2, 25, 10))
                * rng.lognormal(0, 0.4, n), 2)
# survival chance depends on sex, class and age (like the real Titanic)
# the sigmoid 1 / (1 + e^-logit) turns any score into a probability between 0 and 1
logit = 1.2 * (sex == "female") - 0.9 * (pclass - 2) - 0.02 * (age - 30) - 0.4
survived = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame(dict(survived=survived, pclass=pclass, sex=sex, age=age,
                       sibsp=sibsp, parch=parch, fare=fare))
# A leaky column: recorded AFTER the voyage, so it secretly contains the answer
df["lifeboat_assigned"] = np.where(survived == 1, 1, (rng.random(n) < 0.03).astype(int))

# 1. Target-wise EDA: survival rate by group
print("Survival rate by class:")
print(df.groupby("pclass")["survived"].mean().round(2).to_string())
print("\nSurvival rate by sex:")
print(df.groupby("sex")["survived"].mean().round(2).to_string())
print("\nMedian fare by outcome:")
print(df.groupby("survived")["fare"].median().to_string())

# 2. Encoding categorical data
df["sex_label"] = df["sex"].map({"female": 0, "male": 1})          # label encoding
onehot = pd.get_dummies(df["pclass"], prefix="class", dtype=int)  # one-hot
print("\nOne-hot columns:", onehot.columns.tolist())

# 3. Feature engineering
df["family_size"] = df["sibsp"] + df["parch"] + 1                  # interaction
df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 60, 100],
                         labels=["child", "teen", "adult", "senior"])  # binning
df["log_fare"] = np.log1p(df["fare"])                              # log(1 + fare)
print("\nSurvival by age group:")
print(df.groupby("age_group", observed=True)["survived"].mean().round(2).to_string())

# 4. Correlation with the target - and a leakage warning
num = df[["survived", "pclass", "sex_label", "age", "family_size",
          "log_fare", "lifeboat_assigned"]]
corr = num.corr()["survived"].drop("survived").round(2).sort_values(key=abs, ascending=False)
print("\nCorrelation with 'survived':")
print(corr.to_string())
for col, r in corr.items():
    if abs(r) > 0.8:
        print(f"\nWARNING: '{col}' has r = {r} with the target. Is it known at prediction"
              " time? If not, it is DATA LEAKAGE - drop it.")
