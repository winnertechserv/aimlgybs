"""
Chapter 6, Example 3: Pearson Correlation & Feature Selection
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This example builds a synthetic weather dataset with three features (Temperature,
Humidity, Pressure) and a target (Rain). It then computes the Pearson correlation
coefficient between each feature and the target to decide which features to keep. The
output confirms the intuition: Temperature and Humidity strongly predict Rain (very
strong correlation), while Pressure has only a weak relationship (ρ = −0.153), so it
falls below the 0.20 threshold and is dropped. The full correlation matrix also reveals
that Temperature and Humidity are highly correlated with each other (ρ = -0.925),
meaning they carry redundant information — an advanced practitioner might even drop one
of them too.

Run it:           python ex3_pearson_correlation_feature_selection.py
Expected output:  expected_output/ex3_pearson_correlation_feature_selection.txt
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 50
temperature = np.random.uniform(20, 35, n)
humidity    = 120 - 1.5 * temperature + np.random.normal(0, 3, n)
pressure    = np.random.uniform(995, 1040, n)
rain        = -2*temperature + 0.8*humidity + np.random.normal(0, 5, n)

df = pd.DataFrame({'Temperature': temperature, 'Humidity': humidity,
                   'Pressure': pressure, 'Rain': rain})

print("=== Pearson Correlation with Target (Rain) ===\n")
corr_with_rain = df.corr()['Rain'].drop('Rain')
for feat, val in corr_with_rain.items():
    strength = ('None' if abs(val) < 0.1 else 'Weak' if abs(val) < 0.3 else
                'Moderate' if abs(val) < 0.5 else 'Strong' if abs(val) < 0.7
                else 'Very Strong')
    direction = 'positive' if val > 0 else 'negative'
    print(f"  rho(Rain, {feat:<12}) = {val:7.4f}  -> {strength} {direction}")

print()
print("=== Full Correlation Matrix ===")
print(df.corr().round(3).to_string())
print()
# In a real project, compute these correlations on the training split only.
print("=== Feature Selection Decision (drop threshold: |rho| < 0.20) ===")
for feat, val in corr_with_rain.items():
    action = "KEEP" if abs(val) >= 0.20 else "DROP (weak correlation with target)"
    print(f"  {feat:<14}: rho = {val:+.4f}  -> {action}")
