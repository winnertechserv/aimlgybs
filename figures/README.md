# Figures

All 99 figures from the book, grouped by chapter. Every figure is drawn by the Python code in [`src/`](src); run `python figures/src/make_all.py` from the repository root to redraw them all.

| Script | Chapters |
|---|---|
| `src/part1.py` | 1–6 |
| `src/part2.py` | 7–12 |
| `src/part3.py` | 13–24 |
| `src/part4.py` | 25–34 |

## Chapter 1 — AI Foundations & Text Processing

- **Figure 1.1** — [c01_turing.png](png/c01_turing.png): The Turing Test: a person types questions to two hidden partners and must decide which one is the machine.
- **Figure 1.2** — [c01_nlp_map.png](png/c01_nlp_map.png): A map of NLP tasks: some read language, some write it, and some must do both.
- **Figure 1.3** — [c01_qa_pipeline.png](png/c01_qa_pipeline.png): The steps of a question-answering system.

## Chapter 2 — AI Applications

- **Figure 2.1** — [c02_image_tasks.png](png/c02_image_tasks.png): One picture, three tasks: classification gives one label, detection draws a box round each object, and segmentation colours every pixel by what it belongs to.
- **Figure 2.2** — [c02_s2s.png](png/c02_s2s.png): Spoken translation: recognise the words, translate the text, then speak it in the new language.
- **Figure 2.3** — [c02_fraud.png](png/c02_fraud.png): The data behind Example 3: fraudulent transactions (red) are large and far from home, so an anomaly detector can find them without labels.

## Chapter 3 — Paradigms of Machine Learning I

- **Figure 3.1** — [c03_class_vs_reg.png](png/c03_class_vs_reg.png): The two supervised tasks. Left: a classifier splits the space into a child region and an adult region. Right: a regression line predicts salary.
- **Figure 3.2** — [c03_two_phases.png](png/c03_two_phases.png): Every supervised model is built in a training phase and judged in a testing phase on data it has never seen.
- **Figure 3.3** — [c03_clustering.png](png/c03_clustering.png): Unsupervised learning: with no labels at all, K-means still discovers the two natural groups (children and adults).

## Chapter 4 — Paradigms of Machine Learning II

- **Figure 4.1** — [c04_self_training.png](png/c04_self_training.png): The self-training loop: the model labels the unlabelled data itself, keeps only confident guesses, and retrains.
- **Figure 4.2** — [c04_active_curve.png](png/c04_active_curve.png): Results from Example 2: asking the expert about the most uncertain images beats labelling at random at almost every budget.
- **Figure 4.3** — [c04_online.png](png/c04_online.png): From Example 3: when the rule changes at batch 10, the static model collapses while the online model recovers after one batch.
- **Figure 4.4** — [c04_rl_loop.png](png/c04_rl_loop.png): The reinforcement learning loop: the agent acts, and the environment answers with a new state and a reward.

## Chapter 5 — Exploratory Data Analysis

- **Figure 5.1** — [c05_skew.png](png/c05_skew.png): The 'fare' column from Example 1. A few very large values stretch the tail, so the mean sits well above the median.
- **Figure 5.2** — [c05_boxplot.png](png/c05_boxplot.png): A box plot: the box spans Q1 to Q3, the whiskers reach the furthest normal values, and anything beyond the IQR fences is flagged as an outlier.
- **Figure 5.3** — [c05_heatmap.png](png/c05_heatmap.png): Correlations for the passenger data in Example 3. 'lifeboat' is almost perfectly correlated with survival, a warning sign of data leakage.

## Chapter 6 — Data Normalization & Feature Subset Selection

- **Figure 6.1** — [c06_scaling.png](png/c06_scaling.png): The same 80 people before and after scaling. Raw income is in the tens of thousands while age is in tens; after scaling both features are on comparable ranges.
- **Figure 6.2** — [c06_redundant.png](png/c06_redundant.png): 'Total marks' is built from Marks A and Marks B, so it correlates strongly with both and adds no new information.

## Chapter 7 — Regression Methods I

- **Figure 7.1** — [c07_least_squares.png](png/c07_least_squares.png): Least squares on the salary data: the best line minimises the sum of the squared red gaps.
- **Figure 7.2** — [c07_plane.png](png/c07_plane.png): Multiple linear regression on the weather data from Example 2: with two inputs, the fitted model is a flat plane.
- **Figure 7.3** — [c07_gd.png](png/c07_gd.png): Gradient descent on the diabetes data (Example 3): the error falls step by step until it reaches the exact least-squares solution.

## Chapter 8 — Regression Methods II

- **Figure 8.1** — [c08_poly_fits.png](png/c08_poly_fits.png): Example 1: polynomials of degree 1, 2 and 3 fitted to humidity vs temperature, with five new test days.
- **Figure 8.2** — [c08_overfit_panels.png](png/c08_overfit_panels.png): Ten noisy points from a sine wave. M = 9 passes through every point but swings wildly between them: that is overfitting.
- **Figure 8.3** — [c08_rmse_vs_m.png](png/c08_rmse_vs_m.png): Choosing M: always judge on test (or validation) data, because training error falls no matter what.
- **Figure 8.4** — [c08_ridge.png](png/c08_ridge.png): Ridge regression on the same degree-9 model (Example 3): a small penalty tames the wiggles, a large one flattens the curve.

## Chapter 9 — Probabilistic Models for Classification I

- **Figure 9.1** — [c09_knn.png](png/c09_knn.png): K-NN for the second test person in Example 1. Each circle holds the K nearest neighbours; changing K changes the vote.
- **Figure 9.2** — [c09_confusion.png](png/c09_confusion.png): The confusion matrix from Example 2 (cancer screening). Green cells are correct; red cells are the two kinds of mistake.
- **Figure 9.3** — [c09_mahalanobis.png](png/c09_mahalanobis.png): Why Mahalanobis distance disagrees with plain distance for x2: x2 lies just inside the third adult ring, but well outside the narrow child rings.

## Chapter 10 — Probabilistic Models for Classification II

- **Figure 10.1** — [c10_bayes_1d.png](png/c10_bayes_1d.png): Bayes classifier using height alone. Top: how well each height fits each class. Bottom: Bayes rule turns these into class probabilities that add up to 1.
- **Figure 10.2** — [c10_priors.png](png/c10_priors.png): When children are much more common, the boundary (grey lines, where the probability is 0.5) moves to the right, so more borderline people are called children.
- **Figure 10.3** — [c10_gmm.png](png/c10_gmm.png): Example 3: one Gaussian cannot describe class A, which has two clumps; its single wide bell even covers class B. A two-component GMM fits each clump.

## Chapter 11 — Support Vector Machines I

- **Figure 11.1** — [c11_line_sides.png](png/c11_line_sides.png): The sign of g(x) tells you which side of the line a point is on.
- **Figure 11.2** — [c11_bayes_cases.png](png/c11_bayes_cases.png): The three covariance cases. The red line is the Bayes decision boundary; the dotted line joins the two class means.
- **Figure 11.3** — [c11_perceptron.png](png/c11_perceptron.png): The perceptron from Example 2 starting from a deliberately bad line. Every grey line is one update (11 in total) until no point is misclassified.
- **Figure 11.4** — [c11_xor.png](png/c11_xor.png): The XOR pattern from Example 3. Every straight line leaves points on the wrong side, but a nonlinear method separates the classes easily.

## Chapter 12 — Support Vector Machines II

- **Figure 12.1** — [c12_max_margin.png](png/c12_max_margin.png): The maximum-margin hyperplane from Example 1. The shaded street is as wide as possible; only the two circled support vectors touch it.
- **Figure 12.2** — [c12_soft_margin.png](png/c12_soft_margin.png): Soft margin: a small C gives a wide street with many support vectors (circled); a large C squeezes the street to make fewer training errors.
- **Figure 12.3** — [c12_kernel_lift.png](png/c12_kernel_lift.png): The idea behind kernels: rings that no line can split in 2-D become separable by a flat plane once a third feature is added.
- **Figure 12.4** — [c12_rbf.png](png/c12_rbf.png): The same rings classified by a linear SVM and by an RBF-kernel SVM. The kernel lets the SVM draw a circular boundary.

## Chapter 13 — Dimensionality Reduction

- **Figure 13.1** — [c13_pca_axes.png](png/c13_pca_axes.png): PCA finds the direction of greatest spread (PC1) and the next one at a right angle to it (PC2). Projecting onto PC1 alone keeps most of the information.
- **Figure 13.2** — [c13_scree.png](png/c13_scree.png): Cumulative variance for the 8x8 digit images: a small number of components carries almost all of the information in 64 pixels.
- **Figure 13.3** — [c13_pca_vs_lda.png](png/c13_pca_vs_lda.png): PCA ignores labels and keeps the long, wide direction, which mixes the classes. Fisher discriminant analysis (LDA) picks the direction that keeps them apart.

## Chapter 14 — Decision Trees

- **Figure 14.1** — [c14_tree.png](png/c14_tree.png): A small fruit-classification tree. Each box asks one question about one feature; the leaves give the predicted fruit.
- **Figure 14.2** — [c14_impurity.png](png/c14_impurity.png): Three impurity measures for a two-class node. All are zero when the node is pure and largest when the classes are evenly mixed.
- **Figure 14.3** — [c14_depth.png](png/c14_depth.png): A shallow tree draws a few simple boxes. A fully grown tree carves out tiny regions around single points, which is overfitting; Occam's Razor prefers the simpler tree.

## Chapter 15 — Ensemble Methods

- **Figure 15.1** — [c15_bagging.png](png/c15_bagging.png): Bagging: train each classifier on a different bootstrap sample, then let them vote.
- **Figure 15.2** — [c15_boosting.png](png/c15_boosting.png): Boosting trains learners one after another. Examples the previous learner got wrong (red) get more weight, so the next learner focuses on them.
- **Figure 15.3** — [c15_ensemble_size.png](png/c15_ensemble_size.png): Test accuracy on a synthetic 20-feature problem as the ensemble grows. All three ensembles beat a single tree.

## Chapter 16 — Clustering Techniques

- **Figure 16.1** — [c16_kmeans_steps.png](png/c16_kmeans_steps.png): K-means in action: assign every point to its nearest centre, move each centre to the mean of its points, and repeat until nothing changes.
- **Figure 16.2** — [c16_dendrogram.png](png/c16_dendrogram.png): A dendrogram from agglomerative clustering. Cutting it at a chosen height gives the clusters, with no need to pick K in advance.
- **Figure 16.3** — [c16_dbscan.png](png/c16_dbscan.png): Two crescent-shaped clusters. K-means can only cut the space into round regions; density-based DBSCAN follows the shapes.

## Chapter 17 — Multilayer Feedforward Neural Networks I

- **Figure 17.1** — [c17_neuron.png](png/c17_neuron.png): One artificial neuron: weight each input, add them up with a bias, then pass the total through an activation function.
- **Figure 17.2** — [c17_activations.png](png/c17_activations.png): Common activation functions. The threshold jumps; sigmoid and tanh bend smoothly; ReLU is zero for negatives and a straight line for positives.
- **Figure 17.3** — [c17_mlp.png](png/c17_mlp.png): A multilayer feedforward network. Every neuron in one layer connects to every neuron in the next, and signals flow only forward.

## Chapter 18 — MLFFNN II

- **Figure 18.1** — [c18_backprop.png](png/c18_backprop.png): Backpropagation: signals go forward to produce an error, then the error gradient flows backward so every weight knows which way to move.
- **Figure 18.2** — [c18_modes.png](png/c18_modes.png): Training loss per epoch for a simple model. Smaller batches make many more updates per epoch, so the loss falls faster per epoch, though each step is noisier.
- **Figure 18.3** — [c18_local_min.png](png/c18_local_min.png): Gradient descent can settle in a local minimum, a dip that is not the lowest point of the error surface.

## Chapter 19 — Deep Feedforward Neural Networks I

- **Figure 19.1** — [c19_optimizers.png](png/c19_optimizers.png): On a long, narrow valley plain gradient descent creeps along slowly, momentum builds up speed, and Adam takes evenly sized steps in every direction.
- **Figure 19.2** — [c19_loss_curves.png](png/c19_loss_curves.png): Loss per step on the same valley: plain gradient descent lags far behind momentum and Adam.

## Chapter 20 — Deep Feedforward Neural Networks II

- **Figure 20.1** — [c20_dropout.png](png/c20_dropout.png): Dropout: on every training step a different random set of neurons is switched off, so no neuron can rely on any one partner.
- **Figure 20.2** — [c20_batchnorm.png](png/c20_batchnorm.png): Batch normalisation rescales each neuron's outputs over a mini-batch to mean 0 and standard deviation 1, before the learned scale and shift are applied.

## Chapter 21 — Convolutional Neural Networks I

- **Figure 21.1** — [c21_convolution.png](png/c21_convolution.png): Convolution: the kernel sits on the red 3x3 patch, multiplies matching cells and adds them up to give one output value (-2). Then it slides one step and repeats.
- **Figure 21.2** — [c21_pooling.png](png/c21_pooling.png): Max pooling with a 2x2 window and stride 2: each coloured block is replaced by its largest value (in bold).
- **Figure 21.3** — [c21_cnn_arch.png](png/c21_cnn_arch.png): A typical small CNN. With 'same' padding the convolutions keep the width and height, while each pooling layer halves them; the depth grows as more filters are added.

## Chapter 22 — Convolutional Neural Networks II

- **Figure 22.1** — [c22_residual.png](png/c22_residual.png): A residual block. The input skips round the two layers and is added back, so the block only has to learn the change F(x).
- **Figure 22.2** — [c22_vanishing.png](png/c22_vanishing.png): Illustration of the vanishing gradient. If each layer shrinks the gradient (here by a factor of 0.12), early layers get almost nothing; the skip path keeps it close to 1.
- **Figure 22.3** — [c22_transfer.png](png/c22_transfer.png): Transfer learning: keep the feature-extracting layers learned on a big dataset and train only a small new head for your task.

## Chapter 23 — Recurrent Neural Networks I

- **Figure 23.1** — [c23_unrolled.png](png/c23_unrolled.png): An RNN unrolled over four time steps. The same weights U, W and V are reused at every step, and the hidden state h carries memory forward.
- **Figure 23.2** — [c23_lstm.png](png/c23_lstm.png): A simplified LSTM cell. Three gates decide what to erase from, add to, and read out of the cell state, which carries information across many steps.

## Chapter 24 — Recurrent Neural Networks II

- **Figure 24.1** — [c24_seq2seq.png](png/c24_seq2seq.png): An encoder-decoder for translation. The encoder squeezes the sentence into a context vector, and the decoder generates the translation word by word.
- **Figure 24.2** — [c24_embeddings.png](png/c24_embeddings.png): An illustration of word embeddings. Similar words sit close together, and consistent relationships such as man to woman and king to queen point in the same direction.

## Chapter 25 — Transformer I

- **Figure 25.1** — [c25_sdpa.png](png/c25_sdpa.png): Scaled dot-product attention: score each key against the query, scale, turn the scores into weights with softmax, then take the weighted sum of the values.
- **Figure 25.2** — [c25_attention_map.png](png/c25_attention_map.png): An illustration of self-attention weights. Each row shows how much one word draws on every other word; the word 'it' puts most of its weight on 'animal'.
- **Figure 25.3** — [c25_encoder_layer.png](png/c25_encoder_layer.png): One Transformer encoder layer: self-attention then a feedforward network, each wrapped in a residual connection and layer normalisation. The layer is stacked N times.

## Chapter 26 — Transformer II

- **Figure 26.1** — [c26_posenc.png](png/c26_posenc.png): Every row is the position vector for one word position. Early dimensions change quickly from row to row, later ones slowly, so every position gets a unique pattern.
- **Figure 26.2** — [c26_complexity.png](png/c26_complexity.png): Cost per layer with d = 512. Self-attention is cheaper than a recurrent layer for sentences shorter than d, but its cost grows with the square of the length.
- **Figure 26.3** — [c26_bert_gpt.png](png/c26_bert_gpt.png): Which words each position may look at (v = allowed). BERT sees the whole sentence and fills in masked words; GPT sees only earlier words and predicts the next one.

## Chapter 27 — Transformer III

- **Figure 27.1** — [c27_vit.png](png/c27_vit.png): The Vision Transformer idea: cut the image into patches and feed them to a Transformer as if each patch were a word.
- **Figure 27.2** — [c27_rag.png](png/c27_rag.png): Retrieval Augmented Generation: fetch the most relevant passages first, then give them to the LLM with the question so its answer rests on real sources.

## Chapter 28 — Generative Adversarial Networks I

- **Figure 28.1** — [c28_gan.png](png/c28_gan.png): A GAN: the generator turns noise into fakes, the discriminator tries to tell fakes from real data, and each one improves by competing with the other.
- **Figure 28.2** — [c28_training.png](png/c28_training.png): Illustration of GAN training in one dimension. As the generator's distribution moves onto the real one, the best possible discriminator falls to 0.5 everywhere: it can no longer tell them apart.
- **Figure 28.3** — [c28_dcgan.png](png/c28_dcgan.png): A DCGAN generator grows a short noise vector into a full image, using transposed convolutions that double the size at each step.

## Chapter 29 — Generative Adversarial Networks II

- **Figure 29.1** — [c29_cgan.png](png/c29_cgan.png): A conditional GAN: both networks also receive the condition C, so you can ask for a particular kind of output.
- **Figure 29.2** — [c29_cyclegan.png](png/c29_cyclegan.png): CycleGAN uses two generators. Translating a picture there and back should give the original picture, and that cycle loss replaces the need for paired examples.

## Chapter 30 — Applications of Generative AI

- **Figure 30.1** — [c30_ae_vae.png](png/c30_ae_vae.png): Top: an autoencoder squeezes an image into a code and rebuilds it. Bottom: a VAE encodes a mean and spread instead, and samples the code, so new codes give new images.
- **Figure 30.2** — [c30_diffusion.png](png/c30_diffusion.png): The forward diffusion process on a real photo (1,000 steps, linear noise schedule). The model learns to run this in reverse, turning pure noise back into an image.

## Chapter 31 — MLOps

- **Figure 31.1** — [c31_lifecycle.png](png/c31_lifecycle.png): The nine stages of the ML lifecycle. Retraining feeds back into the start, which turns deployment into a loop rather than a one-off task.
- **Figure 31.2** — [c31_drift.png](png/c31_drift.png): Drift: the live data has moved away from the data the model was trained on. The Population Stability Index (PSI) puts a number on how far.

## Chapter 32 — Cloud Deployment

- **Figure 32.1** — [c32_compute.png](png/c32_compute.png): Cloud compute options, from full control (a virtual machine) to almost nothing to manage (serverless).
- **Figure 32.2** — [c32_architecture.png](png/c32_architecture.png): A typical model-serving setup: requests pass through a load balancer to an autoscaling group of model containers, all inside a private network.
- **Figure 32.3** — [c32_autoscaling.png](png/c32_autoscaling.png): Autoscaling over one day. Each instance handles 150 requests per second; the scaler adds or removes instances to keep them near 70% busy, with a minimum of two.

## Chapter 33 — Reinforcement Learning I

- **Figure 33.1** — [c33_agent_env.png](png/c33_agent_env.png): The agent picks an action from its policy; the environment returns the next state and a reward. The goal is the largest total reward over time.
- **Figure 33.2** — [c33_grid.png](png/c33_grid.png): The 4x3 grid world solved by value iteration: the utility of every square and the best action from it. Note how the agent near the -1 square heads away from it rather than risk slipping in.
- **Figure 33.3** — [c33_convergence.png](png/c33_convergence.png): How the utility estimates of four squares settle as value iteration repeats the Bellman update.

## Chapter 34 — Reinforcement Learning II

- **Figure 34.1** — [c34_policy_iter.png](png/c34_policy_iter.png): Policy iteration alternates two steps until the policy stops changing.
- **Figure 34.2** — [c34_cliff.png](png/c34_cliff.png): The greedy paths learned in Example 2. Q-learning hugs the cliff edge; SARSA, which accounts for its own random exploration, keeps well away.
- **Figure 34.3** — [c34_rewards.png](png/c34_rewards.png): Average reward per episode from Example 2 (20 runs). Q-learning keeps falling off the cliff during exploration, so SARSA collects more reward.
- **Figure 34.4** — [c34_actor_critic.png](png/c34_actor_critic.png): Actor-critic: the actor chooses actions, the critic scores them with the TD error, and that score tells the actor which way to improve.
- **Figure 34.5** — [c34_mcts.png](png/c34_mcts.png): The four phases of Monte Carlo Tree Search, repeated many times; the move with the best record at the root is then played.
