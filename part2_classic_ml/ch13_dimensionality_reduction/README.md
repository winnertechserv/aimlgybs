# Chapter 13 — Dimensionality Reduction: PCA & Fisher Discriminant Analysis

**Part II — Classic Machine Learning** · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part2_classic_ml/ch13_dimensionality_reduction/ch13_notebook.ipynb)

Companion code for Chapter 13 of *AI-Machine Learning: Get Your Base Strong*. The book explains the ideas; this folder lets you run them.

## Key ideas

- PCA finds new axes (principal components) ordered by how much variance they capture, and you keep enough to reach about 95%.
- Eigenvalues tell you the variance along each component; eigenvectors give the directions themselves.
- PCA is unsupervised: it never looks at the class labels, so its best direction may be useless for classification.
- FDA (LDA) is supervised and maximizes class separation, which is why its 1-D projection usually classifies better than PCA's.
- Dimensionality reduction is about removing redundancy, not about throwing away data at random.

## Examples

| # | Program | What it shows |
|---|---|---|
| 1 | [`ex1_pca_step_by_step_from_scratch.py`](ex1_pca_step_by_step_from_scratch.py) | This example builds PCA manually without sklearn — step by step, exactly as described in the overview. |
| 2 | [`ex2_pca_with_scikit_learn_on_handwritten_digits.py`](ex2_pca_with_scikit_learn_on_handwritten_digits.py) | This is a real-world PCA example on the famous handwritten digits dataset — the same idea scales to larger 28×28 images. |
| 3 | [`ex3_fisher_discriminant_analysis_fda_lda_vs_pca.py`](ex3_fisher_discriminant_analysis_fda_lda_vs_pca.py) | This is the key example that proves WHY FDA beats PCA for classification tasks. |

Run any example from this folder:

```bash
cd part2_classic_ml/ch13_dimensionality_reduction
python ex1_pca_step_by_step_from_scratch.py
```

The exact output printed in the book is in [`expected_output/`](expected_output). The notebook [`ch13_notebook.ipynb`](ch13_notebook.ipynb) has all three examples with explanations.

### Example 1: PCA Step-by-Step from Scratch

This example builds PCA manually without sklearn — step by step, exactly as described in the overview. We have a 20-sample, 5-feature atmospheric dataset. First we compute the covariance matrix, then perform eigendecomposition to find the principal directions. We sort eigenvalues in descending order (most important direction first) and apply the 95% variance criterion to choose how many dimensions to keep. Notice that PC1 alone captures 99.73% of variance — the data is essentially 1-dimensional despite having 5 features! Finally we project the 5D data into 1D and check the output is uncorrelated.

<details><summary>Expected output</summary>

```text
Dataset shape: (20, 5)  (20 samples, 5 features)

Step 1: Covariance matrix computed (5x5)

Step 2: Eigenvalues and cumulative variance:
  PC1:    342787.18  |  99.73%  |  cum 99.73% <- 95% crossed
  PC2:       790.08  |   0.23%  |  cum 99.96%
  PC3:       130.23  |   0.04%  |  cum 99.99%
  PC4:        16.40  |   0.00%  |  cum 100.00%
  PC5:         2.49  |   0.00%  |  cum 100.00%

Step 3: Keep l = 1 principal component(s) (95% rule)
Step 4: Original shape (20, 5) -> Projected shape (20, 1)

First 5 projected samples:
  Sample 1: [-443.26]
  Sample 2: [-530.7]
  Sample 3: [452.14]
  Sample 4: [-480.82]
  Sample 5: [-375.42]

Step 5: Variance of PC1: 325647.8184
        (equals eigenvalue of PC1 - data is uncorrelated in new space)
```

</details>

### Example 2: PCA with scikit-learn on Handwritten Digits

This is a real-world PCA example on the famous handwritten digits dataset — the same idea scales to larger 28×28 images. Each image is 8×8 = 64 pixels. We use sklearn's PCA to find how many principal components are needed to capture 95% and 99% of the variance. The output shows that just 40 out of 64 dimensions preserve 95% of information — a 37.5% size reduction. The reconstruction error (MSE) is tiny, confirming that the 24 discarded dimensions were mostly noise. This is exactly what the eigenface compression concept shows — you store a few component scores per image instead of all raw pixel values.

<details><summary>Expected output</summary>

```text
Original dataset shape : (1797, 64)
  -> 1797 images, each 8x8 = 64 pixel features

Variance explained by top N components:
  Top   1 PCs -> cumulative variance: 12.03%
  Top   2 PCs -> cumulative variance: 21.59%
  Top   3 PCs -> cumulative variance: 30.04%
  Top   5 PCs -> cumulative variance: 41.40%
  Top  10 PCs -> cumulative variance: 58.87%
  Top  20 PCs -> cumulative variance: 79.31%
  Top  40 PCs -> cumulative variance: 95.08%
  Top  54 PCs -> cumulative variance: 99.08%

95% threshold: keep 40 components
99% threshold: keep 54 components
Reduction: 64 -> 40 dims  (37.5% smaller)

Reduced dataset shape  : (1797, 40)
Reconstruction MSE     : 0.046914
(Small MSE = minimal information lost)
```

</details>

### Example 3: Fisher Discriminant Analysis (FDA/LDA) vs PCA

This is the key example that proves WHY FDA beats PCA for classification tasks. We use sklearn's LinearDiscriminantAnalysis (which builds FDA) and compare it to PCA on the same 10-feature, 2-class dataset. Both reduce to 1 dimension. FDA finds a direction that maximizes class separation (the class means are 1.96 apart after projection). PCA finds the direction of maximum variance — but that direction might not be useful for distinguishing classes. The accuracy comparison confirms it: FDA 1D is nearly as good as the full 10D dataset, while PCA 1D loses large accuracy.

<details><summary>Expected output</summary>

```text
=== FDA vs PCA for Classification ===

Dataset: 300 samples, 10 features, 2 classes

Step 1: FDA projection direction maximizes class separation
  Class +1 mean in FDA space: 1.0191
  Class  0 mean in FDA space: -0.9443
  Separation between means  : 1.9633

Step 2: Classification accuracy comparison (5-NN classifier)
  Original 10D data      : 82.2%
  PCA 1D projection      : 73.3%  (max variance, may miss class info)
  FDA 1D projection      : 80.0%  (max separability)

FDA projection weights per feature:
  Feature  1: +0.0033
  Feature  2: -0.1774
  Feature  3: +0.7178
  Feature  4: -0.3141
  Feature  5: +0.6349
  Feature  6: -0.3084
  Feature  7: +0.1813
  Feature  8: -0.4975
  Feature  9: -1.0280
  Feature 10: -0.2835
```

</details>

## Figures from this chapter

**Figure 13.1.** PCA finds the direction of greatest spread (PC1) and the next one at a right angle to it (PC2). Projecting onto PC1 alone keeps most of the information.

![Figure 13.1](../../figures/png/c13_pca_axes.png)

**Figure 13.2.** Cumulative variance for the 8x8 digit images: a small number of components carries almost all of the information in 64 pixels.

![Figure 13.2](../../figures/png/c13_scree.png)

**Figure 13.3.** PCA ignores labels and keeps the long, wide direction, which mixes the classes. Fisher discriminant analysis (LDA) picks the direction that keeps them apart.

![Figure 13.3](../../figures/png/c13_pca_vs_lda.png)

The code that draws these figures is in [`figures/src`](../../figures/src).

---

[← Chapter 12](../../part2_classic_ml/ch12_support_vector_machines_ii) · [All chapters](../../README.md#chapters) · [Chapter 14 →](../../part2_classic_ml/ch14_decision_trees)
