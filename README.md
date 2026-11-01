# AI-Machine Learning: Get Your Base Strong — Code & Notebooks

[![Check all programs](https://github.com/winnertechserv/aimlgybs/actions/workflows/check.yml/badge.svg)](https://github.com/winnertechserv/aimlgybs/actions/workflows/check.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![Programs: 102](https://img.shields.io/badge/programs-102-orange)
![Figures: 99](https://img.shields.io/badge/figures-99-green)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

This is the companion repository for the book **_AI-Machine Learning: Get Your Base Strong_** by **Abhinav Srivastava**.

The book explains machine learning in plain English, from what AI is to Transformers, generative AI, MLOps and reinforcement learning, across 34 chapters. This repository holds everything in the book that you can **run**: all 102 Python programs, a notebook for every chapter, and the code behind all 99 figures.

## How the book and this repository fit together

| In the book | In this repository |
|---|---|
| Plain-English explanation of every idea | The same ideas as code you can run and change |
| 3 Python programs per chapter, printed with their output | Every program as a `.py` file, with its expected output in `expected_output/` |
| Diagrams with captions | The Python code that draws every diagram, plus the images |
| 20 practice questions per chapter | A notebook per chapter to experiment beyond the questions |
| A fixed printed edition | A living copy: automatically re-tested, with fixes and an [errata list](ERRATA.md) |

**Suggested way to study:** read the chapter in the book, open that chapter's folder here, run the three programs, and compare your output with the book. Then change something (a number, a dataset, a setting), predict what will happen, and run it again. That last step is where most of the learning happens.

## Quick start

**Option 1: no installation.** Click any **Open in Colab** badge in the table below. The notebook opens in your browser with the code and explanations; press *Run all*.

**Option 2: on your own computer** (Python 3.11 or newer):

```bash
git clone https://github.com/winnertechserv/aimlgybs.git
cd aimlgybs
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run one example
cd part2_classic_ml/ch07_regression_methods_i
python ex1_simple_linear_regression_from_scratch.py
```

To check that **every** program still produces the output printed in the book:

```bash
python tools/check_all.py            # all 102 programs
python tools/check_all.py --chapter 7   # just one chapter
```

## Chapters

| Ch. | Topic | Programs | Notebook |
|---|---|---|---|
| **Part I** | **Foundations** | | |
| 1 | [AI Foundations & Text Processing: Pattern Recognition and NLP Tasks](part1_foundations/ch01_ai_foundations_text_processing) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part1_foundations/ch01_ai_foundations_text_processing/ch01_notebook.ipynb) |
| 2 | [AI Applications: Image, Video, Speech & Real-World Domains](part1_foundations/ch02_ai_applications) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part1_foundations/ch02_ai_applications/ch02_notebook.ipynb) |
| 3 | [Paradigms of Machine Learning I: Supervised & Unsupervised Learning](part1_foundations/ch03_paradigms_of_machine_learning_i) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part1_foundations/ch03_paradigms_of_machine_learning_i/ch03_notebook.ipynb) |
| 4 | [Paradigms of Machine Learning II: Semi-Supervised, Active, Self-Supervised, Online, Meta & Reinforcement Learning](part1_foundations/ch04_paradigms_of_machine_learning_ii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part1_foundations/ch04_paradigms_of_machine_learning_ii/ch04_notebook.ipynb) |
| 5 | [Exploratory Data Analysis: Cleaning, Profiling & Feature Engineering](part1_foundations/ch05_exploratory_data_analysis) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part1_foundations/ch05_exploratory_data_analysis/ch05_notebook.ipynb) |
| 6 | [Data Normalization & Feature Subset Selection](part1_foundations/ch06_data_normalization_feature_subset_selection) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part1_foundations/ch06_data_normalization_feature_subset_selection/ch06_notebook.ipynb) |
| **Part II** | **Classic Machine Learning** | | |
| 7 | [Regression Methods I: Simple & Multiple Linear Regression](part2_classic_ml/ch07_regression_methods_i) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part2_classic_ml/ch07_regression_methods_i/ch07_notebook.ipynb) |
| 8 | [Regression Methods II: Polynomial Regression, Overfitting & Ridge Regression](part2_classic_ml/ch08_regression_methods_ii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part2_classic_ml/ch08_regression_methods_ii/ch08_notebook.ipynb) |
| 9 | [Probabilistic Models for Classification I: K-NN, Evaluation Metrics & Reference Templates](part2_classic_ml/ch09_probabilistic_models_for_classification_i) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part2_classic_ml/ch09_probabilistic_models_for_classification_i/ch09_notebook.ipynb) |
| 10 | [Probabilistic Models for Classification II: Bayes Classifier, Naive Bayes & Gaussian Mixture Models](part2_classic_ml/ch10_probabilistic_models_for_classification_ii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part2_classic_ml/ch10_probabilistic_models_for_classification_ii/ch10_notebook.ipynb) |
| 11 | [Support Vector Machines I: Discriminant Functions, Decision Boundaries & the Perceptron](part2_classic_ml/ch11_support_vector_machines_i) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part2_classic_ml/ch11_support_vector_machines_i/ch11_notebook.ipynb) |
| 12 | [Support Vector Machines II: Maximum Margin, Soft Margin, Kernels & Multiclass SVMs](part2_classic_ml/ch12_support_vector_machines_ii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part2_classic_ml/ch12_support_vector_machines_ii/ch12_notebook.ipynb) |
| 13 | [Dimensionality Reduction: PCA & Fisher Discriminant Analysis](part2_classic_ml/ch13_dimensionality_reduction) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part2_classic_ml/ch13_dimensionality_reduction/ch13_notebook.ipynb) |
| 14 | [Decision Trees](part2_classic_ml/ch14_decision_trees) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part2_classic_ml/ch14_decision_trees/ch14_notebook.ipynb) |
| 15 | [Ensemble Methods](part2_classic_ml/ch15_ensemble_methods) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part2_classic_ml/ch15_ensemble_methods/ch15_notebook.ipynb) |
| 16 | [Clustering Techniques](part2_classic_ml/ch16_clustering_techniques) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part2_classic_ml/ch16_clustering_techniques/ch16_notebook.ipynb) |
| **Part III** | **Neural Networks & Deep Learning** | | |
| 17 | [Multilayer Feedforward Neural Networks I](part3_neural_networks/ch17_multilayer_feedforward_neural_networks_i) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part3_neural_networks/ch17_multilayer_feedforward_neural_networks_i/ch17_notebook.ipynb) |
| 18 | [MLFFNN II: Backpropagation & Training](part3_neural_networks/ch18_mlffnn_ii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part3_neural_networks/ch18_mlffnn_ii/ch18_notebook.ipynb) |
| 19 | [Deep Feedforward Neural Networks I: Optimization Methods](part3_neural_networks/ch19_deep_feedforward_neural_networks_i) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part3_neural_networks/ch19_deep_feedforward_neural_networks_i/ch19_notebook.ipynb) |
| 20 | [Deep Feedforward Neural Networks II: Dropout, DropConnect & Normalization](part3_neural_networks/ch20_deep_feedforward_neural_networks_ii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part3_neural_networks/ch20_deep_feedforward_neural_networks_ii/ch20_notebook.ipynb) |
| 21 | [Convolutional Neural Networks I: Convolution, Pooling & CNN Structure](part3_neural_networks/ch21_convolutional_neural_networks_i) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part3_neural_networks/ch21_convolutional_neural_networks_i/ch21_notebook.ipynb) |
| 22 | [Convolutional Neural Networks II: ResNet, Training, Transfer Learning & Applications](part3_neural_networks/ch22_convolutional_neural_networks_ii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part3_neural_networks/ch22_convolutional_neural_networks_ii/ch22_notebook.ipynb) |
| 23 | [Recurrent Neural Networks I: RNNs, LSTM, GRU & BPTT](part3_neural_networks/ch23_recurrent_neural_networks_i) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part3_neural_networks/ch23_recurrent_neural_networks_i/ch23_notebook.ipynb) |
| 24 | [Recurrent Neural Networks II: Embeddings, Seq2Seq & Evaluation Metrics](part3_neural_networks/ch24_recurrent_neural_networks_ii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part3_neural_networks/ch24_recurrent_neural_networks_ii/ch24_notebook.ipynb) |
| **Part IV** | **Transformers & Generative AI** | | |
| 25 | [Transformer I: Attention, Multi-Head Attention & the Encoder-Decoder](part4_transformers_genai/ch25_transformer_i) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part4_transformers_genai/ch25_transformer_i/ch25_notebook.ipynb) |
| 26 | [Transformer II: Position Encoding, Complexity & Pre-Training (BERT, GPT)](part4_transformers_genai/ch26_transformer_ii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part4_transformers_genai/ch26_transformer_ii/ch26_notebook.ipynb) |
| 27 | [Transformer III: Vision Transformers, VQA, Large Language Models & RAG](part4_transformers_genai/ch27_transformer_iii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part4_transformers_genai/ch27_transformer_iii/ch27_notebook.ipynb) |
| 28 | [Generative Adversarial Networks I: Adversarial Training, DCGAN & Evaluation](part4_transformers_genai/ch28_generative_adversarial_networks_i) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part4_transformers_genai/ch28_generative_adversarial_networks_i/ch28_notebook.ipynb) |
| 29 | [Generative Adversarial Networks II: Conditional GANs, CycleGAN & StyleGAN](part4_transformers_genai/ch29_generative_adversarial_networks_ii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part4_transformers_genai/ch29_generative_adversarial_networks_ii/ch29_notebook.ipynb) |
| 30 | [Applications of Generative AI: VAEs, Diffusion Models & LLM Agents](part4_transformers_genai/ch30_applications_of_generative_ai) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part4_transformers_genai/ch30_applications_of_generative_ai/ch30_notebook.ipynb) |
| **Part V** | **Production & Reinforcement Learning** | | |
| 31 | [MLOps: The Machine Learning Lifecycle in Production](part5_production_rl/ch31_mlops) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part5_production_rl/ch31_mlops/ch31_notebook.ipynb) |
| 32 | [Cloud Deployment: Compute, Containers, Scaling & Serverless](part5_production_rl/ch32_cloud_deployment) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part5_production_rl/ch32_cloud_deployment/ch32_notebook.ipynb) |
| 33 | [Reinforcement Learning I: MDPs, Bellman Equation & Value Iteration](part5_production_rl/ch33_reinforcement_learning_i) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part5_production_rl/ch33_reinforcement_learning_i/ch33_notebook.ipynb) |
| 34 | [Reinforcement Learning II: Policy Iteration, SARSA, Deep RL & Actor-Critic](part5_production_rl/ch34_reinforcement_learning_ii) | 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/winnertechserv/aimlgybs/blob/main/part5_production_rl/ch34_reinforcement_learning_ii/ch34_notebook.ipynb) |

## Repository layout

```
aimlgybs/
├── part1_foundations/ … part5_production_rl/   one folder per part
│   └── chNN_topic/
│       ├── README.md            key ideas, what each example shows, expected outputs, the chapter's figures
│       ├── exN_*.py             the programs, exactly as printed in the book
│       ├── expected_output/     the exact output printed in the book
│       └── chNN_notebook.ipynb  the same programs as a notebook (opens in Colab)
├── figures/
│   ├── png/                     all 99 figures from the book
│   └── src/                     the code that draws them (python figures/src/make_all.py)
├── tools/check_all.py           runs every program and compares with the book
├── requirements.txt             exact library versions used for the book's outputs
└── ERRATA.md                    corrections to the printed book
```

## About the outputs and library versions

Every output in the book was produced with the versions in [`requirements.txt`](requirements.txt), and the automated check re-runs all 102 programs on every change. All programs use fixed random seeds, so you should see exactly the same numbers. With newer library versions, a few programs that train neural networks or use random sampling may differ in the last digits. The lesson of each example does not change.

No dataset needs downloading: the programs either build their data in code or use datasets bundled with scikit-learn.

## Found a mistake or have a question?

- **Mistake in the book or code?** Open an [issue](https://github.com/winnertechserv/aimlgybs/issues/new/choose) using the *Errata* template. Confirmed fixes are listed in [ERRATA.md](ERRATA.md).
- **Question about a concept or an example?** Open an issue using the *Question* template.

## License

The code and figures in this repository are released under the [MIT License](LICENSE). You may use them, with credit. The text of the book is not part of this repository and remains © Abhinav Srivastava.
