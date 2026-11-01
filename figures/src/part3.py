from common import *
from sklearn.datasets import load_digits, make_moons, make_blobs, make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram

def regions(ax, model, X, y, lim=None, cols=('#EAF2FB', PALE)):
    x0, x1 = (X[:, 0].min() - .5, X[:, 0].max() + .5); y0, y1 = (X[:, 1].min() - .5, X[:, 1].max() + .5)
    G = np.meshgrid(np.linspace(x0, x1, 250), np.linspace(y0, y1, 250)); Z = model.predict(np.c_[G[0].ravel(), G[1].ravel()]).reshape(G[0].shape)
    ax.contourf(*G, Z, levels=[-0.5, 0.5, 1.5], colors=list(cols)); ax.scatter(*X[y == 0].T, s=10, c=NAVY); ax.scatter(*X[y == 1].T, s=10, c=ORANGE)

# ---------------- Chapter 13 ----------------
rng = np.random.default_rng(0); Xp = rng.multivariate_normal([0, 0], [[3, 2.2], [2.2, 2.2]], 200)
pca = PCA().fit(Xp)
fig, ax = plt.subplots(figsize=(5.4, 4.4)); ax.scatter(*Xp.T, s=10, c=NAVY, alpha=0.5)
for vec, var, c, lab in zip(pca.components_, pca.explained_variance_, [ORANGE, RED], ['PC1', 'PC2']):
    v = vec * 2.2 * np.sqrt(var); ax.annotate('', xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=c, lw=2.5)); ax.text(*(v * 1.12), lab, color=c, fontweight='bold')
ax.set_aspect('equal'); ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
ax.set_title(f'PC1 keeps {100 * pca.explained_variance_ratio_[0]:.0f}% of the variance')
save(fig, 'c13_pca_axes.png')
reg(13, "Principal Component Analysis is the standard answer", 'c13_pca_axes.png', 'PCA finds the direction of greatest spread (PC1) and the next one at a right angle to it (PC2). Projecting onto PC1 alone keeps most of the information.')

Xd = load_digits().data; cum = np.cumsum(PCA().fit(Xd).explained_variance_ratio_); k95 = np.argmax(cum >= 0.95) + 1
fig, ax = plt.subplots(figsize=(6.4, 3.4)); ax.plot(range(1, 65), cum, c=NAVY, lw=2); ax.axhline(0.95, c=ORANGE, ls='--'); ax.axvline(k95, c=ORANGE, ls='--')
ax.text(k95 + 1, 0.6, f'{k95} components keep 95%', color=ORANGE); ax.set_xlabel('Number of principal components'); ax.set_ylabel('Cumulative variance kept'); ax.set_title('Digits: 64 pixels, but far fewer directions matter')
save(fig, 'c13_scree.png')
reg(13, "The recipe is short", 'c13_scree.png', 'Cumulative variance for the 8x8 digit images: a small number of components carries almost all of the information in 64 pixels.')

rng = np.random.default_rng(1); a = rng.multivariate_normal([0, 0], [[6, 0], [0, 0.25]], 150); b = rng.multivariate_normal([0, 1.6], [[6, 0], [0, 0.25]], 150)
Xl = np.vstack([a, b]); yl = np.r_[np.zeros(150), np.ones(150)]
p1 = PCA(1).fit_transform(Xl)[:, 0]; l1 = LinearDiscriminantAnalysis(n_components=1).fit_transform(Xl, yl)[:, 0]
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
axs[0].scatter(*a.T, s=8, c=NAVY); axs[0].scatter(*b.T, s=8, c=ORANGE); axs[0].set_title('Two classes, long and thin')
axs[0].annotate('', xy=(7.5, -2.3), xytext=(-7.5, -2.3), arrowprops=dict(arrowstyle='<->', color=GRAY, lw=2)); axs[0].text(-7.4, -2.05, 'PCA direction (most spread)', color=GRAY, fontsize=8.5)
axs[0].annotate('', xy=(8.6, 3.2), xytext=(8.6, -1.6), arrowprops=dict(arrowstyle='<->', color=GREEN, lw=2)); axs[0].text(8.0, 3.4, 'LDA direction', color=GREEN, fontsize=8.5)
axs[0].set_ylim(-2.8, 4); axs[0].set_xlim(-9, 10)
for ax_, v, t in [(axs[1], p1, 'Projected onto PCA: classes mixed'), (axs[2], l1, 'Projected onto LDA: classes apart')]:
    ax_.hist(v[yl == 0], bins=30, color=NAVY, alpha=0.6); ax_.hist(v[yl == 1], bins=30, color=ORANGE, alpha=0.6); ax_.set_title(t); ax_.set_yticks([])
fig.tight_layout(); save(fig, 'c13_pca_vs_lda.png')
reg(13, "There is one catch, and it matters", 'c13_pca_vs_lda.png', 'PCA ignores labels and keeps the long, wide direction, which mixes the classes. Fisher discriminant analysis (LDA) picks the direction that keeps them apart.')

# ---------------- Chapter 14 ----------------
# The same 12 fruits and the same tree as Chapter 14, Example 2
Xf = np.array([[0,2,0,0],[0,2,0,0],[2,1,0,1],[0,1,0,0],[2,1,0,1],[1,1,0,0],[1,2,0,0],[1,0,1,0],[1,0,1,0],[2,0,0,1],[2,0,0,0],[2,0,1,0]])
fruit = np.array([6,6, 0,0,0, 5,4, 1,1, 2, 3,3]); names = ['Apple','Banana','Cherry','Grape','Grapefruit','Lemon','Watermelon']
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42).fit(Xf, fruit)
fig, ax = plt.subplots(figsize=(11, 5.6))
plot_tree(tree, feature_names=['Color', 'Size', 'Shape', 'Taste'], class_names=names,
          filled=True, impurity=False, proportion=False, rounded=True, fontsize=8, label='none', ax=ax)
ax.set_title('A decision tree is a flowchart of questions\n(Color: 0 green, 1 yellow, 2 red;  Size: 0 small, 1 medium, 2 big;  Taste: 0 not sweet, 1 sweet)', color=NAVY, fontweight='bold', fontsize=10)
save(fig, 'c14_tree.png')
reg(14, "Decision Trees are the go-to", 'c14_tree.png', 'A small fruit-classification tree. Each box asks one question about one feature; the leaves give the predicted fruit.')

p = np.linspace(0.001, 0.999, 300)
ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p)); gini = 2 * p * (1 - p); mis = 1 - np.maximum(p, 1 - p)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(p, ent, c=NAVY, lw=2, label='Entropy'); ax.plot(p, gini, c=ORANGE, lw=2, label='Gini'); ax.plot(p, mis, c=TEAL, lw=2, ls='--', label='Misclassification')
ax.set_xlabel('Share of class 1 at the node (P)'); ax.set_ylabel('Impurity'); ax.legend(frameon=False); ax.set_title('Impurity is zero for a pure node, highest at 50/50')
save(fig, 'c14_impurity.png')
reg(14, "Entropy Impurity is the standard", 'c14_impurity.png', 'Three impurity measures for a two-class node. All are zero when the node is pure and largest when the classes are evenly mixed.')

Xm, ym = make_moons(300, noise=0.3, random_state=0)
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6))
for ax_, d in zip(axs, (3, None)):
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xm, ym); regions(ax_, t, Xm, ym)
    ax_.set_title(f'max_depth = {d}: {t.get_n_leaves()} leaves' + (' (simple)' if d else ' (overfits)'))
fig.tight_layout(); save(fig, 'c14_depth.png')
reg(14, "Occam's Razor", 'c14_depth.png', "A shallow tree draws a few simple boxes. A fully grown tree carves out tiny regions around single points, which is overfitting; Occam's Razor prefers the simpler tree.")

# ---------------- Chapter 15 ----------------
fig, ax = diagram(9, 3.4, ylim=(0, 4))
box(ax, 1.1, 2, 'Training\nset (N)', 1.5, 1.0, fc=TEAL, ec=TEAL, tc='white', bold=True)
for i, yy in enumerate([3.3, 2.0, 0.7]):
    box(ax, 3.5, yy, f'Bootstrap\nsample {i + 1}', 1.6, 0.8); box(ax, 5.9, yy, f'Classifier {i + 1}', 1.6, 0.7, fc=PALE, ec=ORANGE)
    arrow(ax, (1.85, 2), (2.7, yy)); arrow(ax, (4.3, yy), (5.1, yy)); arrow(ax, (6.7, yy), (7.65, 2))
box(ax, 8.4, 2, 'Majority\nvote', 1.4, 1.0, fc=ORANGE, ec=ORANGE, tc='white', bold=True)
ax.text(3.5, 0.05, 'each sample: N draws WITH replacement', fontsize=8, color=GRAY, ha='center')
save(fig, 'c15_bagging.png')
reg(15, "Bagging is short for", 'c15_bagging.png', 'Bagging: train each classifier on a different bootstrap sample, then let them vote.')

fig, ax = diagram(9, 2.8, ylim=(0, 3.2))
for i in range(3):
    x0 = 1.2 + i * 3.1; box(ax, x0, 2.1, f'Round {i + 1}\nweak learner', 1.7, 0.8, fc=PALE, ec=ORANGE)
    for j in range(6):
        big = (i == 1 and j in (1, 4)) or (i == 2 and j in (2, 5))
        ax.add_patch(Circle((x0 - 0.75 + j * 0.3, 0.8), 0.08 + (0.07 if big else 0), color=RED if big else NAVY))
    if i < 2: arrow(ax, (x0 + 0.9, 2.1), (x0 + 2.2, 2.1), 'up-weight\nmistakes', toff=(0, 0.55))
ax.text(4.3, 0.2, 'examples (bigger dot = more weight)', fontsize=8, color=GRAY, ha='center')
save(fig, 'c15_boosting.png')
reg(15, "AdaBoost, short for adaptive", 'c15_boosting.png', 'Boosting trains learners one after another. Examples the previous learner got wrong (red) get more weight, so the next learner focuses on them.')

X, y = make_classification(1000, n_features=20, n_informative=8, random_state=0); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
ns = [1, 5, 10, 25, 50, 100, 200]; single = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte)
bag = [BaggingClassifier(DecisionTreeClassifier(), n_estimators=n, random_state=0).fit(Xtr, ytr).score(Xte, yte) for n in ns]
rf = [RandomForestClassifier(n_estimators=n, random_state=0).fit(Xtr, ytr).score(Xte, yte) for n in ns]
ada = [AdaBoostClassifier(n_estimators=n, random_state=0).fit(Xtr, ytr).score(Xte, yte) for n in ns]
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.axhline(single, c=GRAY, ls='--', label='Single decision tree')
for v, c, m, lab in [(bag, NAVY, 'o', 'Bagging'), (rf, GREEN, 's', 'Random forest'), (ada, ORANGE, '^', 'AdaBoost')]: ax.plot(ns, v, marker=m, c=c, label=lab)
ax.set_xscale('log'); ax.set_xlabel('Number of classifiers in the ensemble (log scale)'); ax.set_ylabel('Test accuracy'); ax.legend(frameon=False, fontsize=8.5); ax.set_title('More voters, better accuracy (up to a point)')
save(fig, 'c15_ensemble_size.png')
reg(15, "AdaBoost, short for adaptive", 'c15_ensemble_size.png', 'Test accuracy on a synthetic 20-feature problem as the ensemble grows. All three ensembles beat a single tree.')

# ---------------- Chapter 16 ----------------
Xb, _ = make_blobs(300, centers=3, cluster_std=0.9, random_state=7); init = Xb[[0, 1, 2]]
fig, axs = plt.subplots(1, 4, figsize=(12, 3.1)); cols = np.array([NAVY, ORANGE, TEAL]); cen = init.copy()
for k, ax_ in enumerate(axs):
    lab = np.argmin(((Xb[:, None] - cen) ** 2).sum(2), 1)
    ax_.scatter(*Xb.T, s=8, c=cols[lab] if k else GRAY); ax_.scatter(*cen.T, marker='*', s=260, c='black')
    ax_.set_title(['Step 1: pick K = 3 centers', 'Assign each point', 'Move centers, reassign', 'Converged'][k]); ax_.set_xticks([]); ax_.set_yticks([])
    for _ in range(1 if k < 3 else 10):
        if k: cen = np.array([Xb[lab == j].mean(0) for j in range(3)]); lab = np.argmin(((Xb[:, None] - cen) ** 2).sum(2), 1)
fig.tight_layout(); save(fig, 'c16_kmeans_steps.png')
reg(16, "K-Means works like this", 'c16_kmeans_steps.png', 'K-means in action: assign every point to its nearest center, move each center to the mean of its points, and repeat until nothing changes.')

rng = np.random.default_rng(3); Xh = np.vstack([rng.normal(c, 0.4, (6, 2)) for c in ([0, 0], [3, 3], [0, 4])])
fig, ax = plt.subplots(figsize=(7, 3.4)); dendrogram(linkage(Xh, 'average'), ax=ax, color_threshold=2.5, above_threshold_color=GRAY)
ax.axhline(2.5, c=RED, ls='--'); ax.text(1, 2.65, 'cut here -> 3 clusters', color=RED, fontsize=9); ax.set_xlabel('Data points'); ax.set_ylabel('Merge distance'); ax.set_title('A dendrogram records every merge')
save(fig, 'c16_dendrogram.png')
reg(16, "Agglomerative hierarchical clustering", 'c16_dendrogram.png', 'A dendrogram from agglomerative clustering. Cutting it at a chosen height gives the clusters, with no need to pick K in advance.')

Xn, _ = make_moons(400, noise=0.06, random_state=0)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for ax_, (name, lab) in zip(axs, [('K-means (K = 2)', KMeans(2, n_init=10, random_state=0).fit_predict(Xn)), ('DBSCAN', DBSCAN(eps=0.2).fit_predict(Xn))]):
    ax_.scatter(*Xn.T, s=8, c=np.array([NAVY, ORANGE, GRAY])[lab]); ax_.set_title(name); ax_.set_xticks([]); ax_.set_yticks([])
fig.tight_layout(); save(fig, 'c16_dbscan.png')
reg(16, "There are three categories of clustering", 'c16_dbscan.png', 'Two crescent-shaped clusters. K-means can only cut the space into round regions; density-based DBSCAN follows the shapes.')

# ---------------- Chapter 17 ----------------
fig, ax = diagram(8.4, 3.2, ylim=(0, 4))
for i, yy in enumerate([3.3, 2.0, 0.7]):
    ax.add_patch(Circle((0.9, yy), 0.35, color=LIGHT, ec=NAVY)); ax.text(0.9, yy, f'x{i + 1}', ha='center', va='center')
    arrow(ax, (1.25, yy), (3.9, 2.0), f'w{i + 1}', toff=(-0.4, 0.25 if yy > 2 else -0.25 if yy < 2 else 0.2))
ax.add_patch(Circle((4.4, 2.0), 0.55, color=PALE, ec=ORANGE, lw=1.5)); ax.text(4.4, 2.0, 'Σ + b', ha='center', va='center', fontweight='bold')
arrow(ax, (4.95, 2.0), (6.1, 2.0), 'a'); box(ax, 6.9, 2.0, 'activation\nf(a)', 1.5, 0.9, fc=TEAL, ec=TEAL, tc='white', bold=True)
arrow(ax, (7.65, 2.0), (8.9, 2.0)); ax.text(9.3, 2.0, 'output\ns = f(a)', va='center', fontsize=9)
ax.text(4.4, 0.6, 'a = w1x1 + w2x2 + w3x3 + b', ha='center', fontsize=9, color=NAVY)
save(fig, 'c17_neuron.png')
reg(17, "The design is loosely borrowed", 'c17_neuron.png', 'One artificial neuron: weight each input, add them up with a bias, then pass the total through an activation function.')

a = np.linspace(-5, 5, 400)
fig, axs = plt.subplots(1, 4, figsize=(11, 2.6), sharex=True)
for ax_, (f, t) in zip(axs, [((a >= 0).astype(float), 'Threshold (McCulloch-Pitts)'), (1 / (1 + np.exp(-a)), 'Logistic (sigmoid)'), (np.tanh(a), 'Tanh'), (np.maximum(0, a), 'ReLU')]):
    ax_.plot(a, f, c=NAVY, lw=2); ax_.axhline(0, c=GRAY, lw=0.5); ax_.axvline(0, c=GRAY, lw=0.5); ax_.set_title(t); ax_.set_xlabel('a')
fig.tight_layout(); save(fig, 'c17_activations.png')
reg(17, "Sigmoidal activations solve that", 'c17_activations.png', 'Common activation functions. The threshold jumps; sigmoid and tanh bend smoothly; ReLU is zero for negatives and a straight line for positives.')

fig, ax = diagram(8, 3.6, ylim=(0, 4.4))
layers = [(1.2, 3, 'Input layer\n(linear)', LIGHT), (4.2, 4, 'Hidden layer\n(nonlinear)', PALE), (7.2, 2, 'Output layer', '#E3F1EC')]
pos = []
for x, n, t, fc in layers:
    ys = np.linspace(3.6 - (4 - n) * 0.45, 0.9 + (4 - n) * 0.45, n); pos.append([(x, yy) for yy in ys]); ax.text(x, 0.15, t, ha='center', fontsize=9, color=NAVY)
for L in range(2):
    for p1 in pos[L]:
        for p2 in pos[L + 1]: ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=GRAY, lw=0.7, zorder=1)
for (x, n, t, fc), ps in zip(layers, pos):
    for p_ in ps: ax.add_patch(Circle(p_, 0.27, fc=fc, ec=NAVY, lw=1.3, zorder=2))
save(fig, 'c17_mlp.png')
reg(17, "Putting it together, an MLFFNN", 'c17_mlp.png', 'A multilayer feedforward network. Every neuron in one layer connects to every neuron in the next, and signals flow only forward.')

# ---------------- Chapter 18 ----------------
fig, ax = diagram(8.6, 3.0, ylim=(0, 3.6))
for i, t in enumerate(['Input x', 'Hidden\nlayer', 'Output\nlayer', 'Error E']):
    box(ax, 1.1 + i * 2.6, 1.8, t, 1.6, 0.9, fc=RED if i == 3 else LIGHT, ec=RED if i == 3 else NAVY, tc='white' if i == 3 else 'black', bold=i == 3)
for i in range(3):
    arrow(ax, (1.95 + i * 2.6, 2.05), (2.85 + i * 2.6, 2.05), color=TEAL)
    arrow(ax, (2.85 + i * 2.6, 1.55), (1.95 + i * 2.6, 1.55), color=ORANGE)
ax.text(4.95, 3.1, 'Forward pass: compute outputs', ha='center', color=TEAL, fontweight='bold'); ax.text(4.95, 0.55, 'Backward pass: send dE/dw back, layer by layer (chain rule)', ha='center', color=ORANGE, fontweight='bold')
save(fig, 'c18_backprop.png')
reg(18, "Forward Pass:", 'c18_backprop.png', 'Backpropagation: signals go forward to produce an error, then the error gradient flows backward so every weight knows which way to move.')

rng = np.random.default_rng(0); Xs = rng.normal(0, 1, (512, 2)); ys = (Xs[:, 0] + 0.5 * Xs[:, 1] + rng.normal(0, 0.5, 512) > 0).astype(float)
def train(bs, epochs=30, lr=0.1):
    w = np.zeros(3); A = np.column_stack([np.ones(512), Xs]); out = []; r = np.random.default_rng(1)
    for e in range(epochs):
        idx = r.permutation(512)
        for s0 in range(0, 512, bs):
            b = idx[s0:s0 + bs]; p = 1 / (1 + np.exp(-A[b] @ w)); w -= lr * A[b].T @ (p - ys[b]) / len(b)
        p = np.clip(1 / (1 + np.exp(-A @ w)), 1e-9, 1 - 1e-9); out.append(-np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p)))
    return out
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for bs, c, lab in [(512, NAVY, 'Batch mode (all 512)'), (32, ORANGE, 'Mini-batch (32)'), (1, TEAL, 'Pattern mode / SGD (1)')]: ax.plot(range(1, 31), train(bs), c=c, lw=2, label=lab)
ax.set_xlabel('Epoch'); ax.set_ylabel('Training loss'); ax.legend(frameon=False); ax.set_title('Same data, three update schedules')
save(fig, 'c18_modes.png')
reg(18, "There are three modes of learning", 'c18_modes.png', 'Training loss per epoch for a simple model. Smaller batches make many more updates per epoch, so the loss falls faster per epoch, though each step is noisier.')

w = np.linspace(-3, 3, 400); E = 0.3 * w ** 4 - 1.2 * w ** 2 + 0.5 * w + 2
fig, ax = plt.subplots(figsize=(6.2, 3.2)); ax.plot(w, E, c=NAVY, lw=2)
from scipy.signal import argrelmin
for i in argrelmin(E)[0]:
    glob_ = E[i] == E[argrelmin(E)[0]].min(); ax.scatter(w[i], E[i], c=GREEN if glob_ else RED, s=60, zorder=3)
    ax.text(w[i], E[i] - 0.45, 'global minimum' if glob_ else 'local minimum', ha='center', color=GREEN if glob_ else RED)
ax.set_xlabel('Weight w'); ax.set_ylabel('Error E(w)'); ax.set_title('A non-convex error surface'); ax.set_ylim(E.min() - 0.8, 5)
save(fig, 'c18_local_min.png')
reg(18, "Local Minima Problem", 'c18_local_min.png', 'Gradient descent can settle in a local minimum, a dip that is not the lowest point of the error surface.')

# ---------------- Chapter 19 ----------------
def grad(p): return np.array([0.2 * p[0], 4 * p[1]])
def run(kind, steps=60, lr=0.2):
    p = np.array([-4.5, 1.5]); v = np.zeros(2); s = np.zeros(2); m = np.zeros(2); path = [p.copy()]
    for t in range(1, steps + 1):
        g = grad(p)
        if kind == 'GD': p = p - 0.45 * lr * g
        elif kind == 'Momentum': v = 0.9 * v - 0.45 * lr * g; p = p + v
        elif kind == 'Adam':
            m = 0.9 * m + 0.1 * g; s = 0.999 * s + 0.001 * g * g; mh = m / (1 - 0.9 ** t); sh = s / (1 - 0.999 ** t); p = p - 0.3 * mh / (np.sqrt(sh) + 1e-8)
        path.append(p.copy())
    return np.array(path)
G = np.meshgrid(np.linspace(-5, 1.5, 200), np.linspace(-2, 2, 200)); Z = 0.1 * G[0] ** 2 + 2 * G[1] ** 2
fig, ax = plt.subplots(figsize=(7, 3.6)); ax.contour(*G, Z, levels=15, colors=GRAY, linewidths=0.6)
for k, c in [('GD', RED), ('Momentum', NAVY), ('Adam', ORANGE)]: P_ = run(k); ax.plot(*P_.T, 'o-', ms=2.5, c=c, lw=1.4, label=k)
ax.scatter(0, 0, marker='*', s=200, c=GREEN, zorder=4, label='minimum'); ax.legend(frameon=False, fontsize=8.5, loc='lower right'); ax.set_xlabel('w1'); ax.set_ylabel('w2')
ax.set_title('A long narrow valley: 60 steps from the same start')
save(fig, 'c19_optimizers.png')
reg(19, "Generalized Delta Rule with Momentum", 'c19_optimizers.png', 'On a long, narrow valley plain gradient descent creeps along slowly, momentum builds up speed, and Adam takes evenly sized steps in every direction.')

fig, ax = plt.subplots(figsize=(6.2, 3.3))
for k, c in [('GD', RED), ('Momentum', NAVY), ('Adam', ORANGE)]:
    P_ = run(k, 150); ax.plot(0.1 * P_[:, 0] ** 2 + 2 * P_[:, 1] ** 2 + 1e-12, c=c, lw=2, label=k)
ax.set_yscale('log'); ax.set_xlabel('Step'); ax.set_ylabel('Loss (log scale)'); ax.legend(frameon=False); ax.set_title('Loss on the same valley')
save(fig, 'c19_loss_curves.png')
reg(19, "Adam (Adaptive Moments)", 'c19_loss_curves.png', 'Loss per step on the same valley: plain gradient descent lags far behind momentum and Adam.')

# ---------------- Chapter 20 ----------------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
rng = np.random.default_rng(4); drop = {(1, 1), (1, 3), (2, 0)}
for ax_, use in zip(axs, (False, True)):
    ax_.axis('off'); ax_.set_xlim(0, 4); ax_.set_ylim(-0.3, 4.3); xs_ = [0.5, 2, 3.5]; ns_ = [4, 5, 2]
    P = [[(xs_[l], 3.8 - i * (3.6 / (ns_[l] - 1)) if ns_[l] > 1 else 2) for i in range(ns_[l])] for l in range(3)]
    for l in range(2):
        for i, p1 in enumerate(P[l]):
            for j, p2 in enumerate(P[l + 1]):
                off = use and ((l, i) in drop or (l + 1, j) in drop)
                ax_.plot([p1[0], p2[0]], [p1[1], p2[1]], c=GRAY, lw=0.6, alpha=0.15 if off else 0.8)
    for l in range(3):
        for i, p_ in enumerate(P[l]):
            off = use and (l, i) in drop
            ax_.add_patch(Circle(p_, 0.2, fc='white' if off else LIGHT, ec=GRAY if off else NAVY, ls='--' if off else '-', zorder=3))
            if off: ax_.text(*p_, 'x', ha='center', va='center', color=RED, fontweight='bold', zorder=4)
    ax_.set_title('Full network' if not use else 'With dropout: random units switched off', color=NAVY, fontweight='bold')
save(fig, 'c20_dropout.png')
reg(20, "Start with dropout", 'c20_dropout.png', 'Dropout: on every training step a different random set of neurons is switched off, so no neuron can rely on any one partner.')

rng = np.random.default_rng(0); act = rng.normal(0, 1, (64, 3)) * np.array([[3, 12, 0.9]]) + np.array([[7, 10, 31]])
bn = (act - act.mean(0)) / act.std(0)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2)); cs = [NAVY, ORANGE, TEAL]
for j in range(3): axs[0].hist(act[:, j], bins=15, color=cs[j], histtype='step', lw=2, label=f'neuron {j + 1}'); axs[1].hist(bn[:, j], bins=12, color=cs[j], histtype='step', lw=2)
axs[0].set_title('Before: each neuron on its own scale'); axs[1].set_title('After batch normalization: mean 0, std 1'); axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); save(fig, 'c20_batchnorm.png')
reg(20, "Batch normalization is the one", 'c20_batchnorm.png', "Batch normalization rescales each neuron's outputs over a mini-batch to mean 0 and standard deviation 1, before the learned scale and shift are applied.")

# ---------------- Chapter 21 ----------------
img = np.array([[1, 0, 2, 1, 0], [0, 1, 3, 1, 1], [2, 1, 0, 0, 2], [1, 3, 1, 2, 0], [0, 1, 2, 1, 1]]); ker = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
out = np.array([[(img[i:i + 3, j:j + 3] * ker).sum() for j in range(3)] for i in range(3)])
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), gridspec_kw=dict(width_ratios=[5, 3, 3]))
def grid(ax_, M, title, hl=None, cmap='Blues'):
    ax_.imshow(M, cmap=cmap, alpha=0.35); ax_.set_xticks([]); ax_.set_yticks([]); ax_.set_title(title)
    for (i, j), v in np.ndenumerate(M): ax_.text(j, i, str(v), ha='center', va='center', fontsize=11)
    if hl: ax_.add_patch(Rectangle((hl[1] - 0.5, hl[0] - 0.5), hl[2], hl[2], fill=False, ec=RED, lw=2.5))
grid(axs[0], img, '5x5 input', (0, 0, 3)); grid(axs[1], ker, '3x3 kernel', cmap='Oranges'); grid(axs[2], out, '3x3 feature map', (0, 0, 1), cmap='Greens')
fig.tight_layout(); save(fig, 'c21_convolution.png')
reg(21, "The big idea is the convolution", 'c21_convolution.png', f'Convolution: the kernel sits on the red 3x3 patch, multiplies matching cells and adds them up to give one output value ({out[0, 0]}). Then it slides one step and repeats.')

fm = np.array([[1, 3, 2, 1], [4, 6, 5, 0], [3, 1, 1, 2], [0, 2, 7, 4]]); mp = fm.reshape(2, 2, 2, 2).max(axis=(1, 3))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3), gridspec_kw=dict(width_ratios=[4, 2]))
cmap = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
axs[0].imshow(cmap, cmap='Pastel1'); axs[1].imshow(np.array([[0, 1], [2, 3]]), cmap='Pastel1')
for (i, j), v in np.ndenumerate(fm): axs[0].text(j, i, v, ha='center', va='center', fontsize=12, fontweight='bold' if v == mp[i // 2, j // 2] else 'normal')
for (i, j), v in np.ndenumerate(mp): axs[1].text(j, i, v, ha='center', va='center', fontsize=12, fontweight='bold')
for a_, t in zip(axs, ['4x4 feature map', '2x2 after max pooling']): a_.set_xticks([]); a_.set_yticks([]); a_.set_title(t)
fig.tight_layout(); save(fig, 'c21_pooling.png')
reg(21, "After convolution comes pooling", 'c21_pooling.png', 'Max pooling with a 2x2 window and stride 2: each colored block is replaced by its largest value (in bold).')

fig, ax = diagram(10, 2.6, ylim=(0, 3))
stages = [('Image\n32x32x3', LIGHT), ('Conv + ReLU\n32x32x16', PALE), ('Max pool\n16x16x16', '#E3F1EC'), ('Conv + ReLU\n16x16x32', PALE), ('Max pool\n8x8x32', '#E3F1EC'), ('Fully\nconnected', LIGHT), ('Softmax\n10 classes', ORANGE)]
for i, (t, fc) in enumerate(stages):
    box(ax, 0.75 + i * 1.42, 1.5, t, 1.25, 1.1, fc=fc, ec=NAVY, fs=8, bold=i == 6, tc='white' if i == 6 else 'black')
    if i: arrow(ax, (0.75 + (i - 1) * 1.42 + 0.63, 1.5), (0.75 + i * 1.42 - 0.63, 1.5))
save(fig, 'c21_cnn_arch.png')
reg(21, "The size of a feature map", 'c21_cnn_arch.png', "A typical small CNN. With 'same' padding the convolutions keep the width and height, while each pooling layer halves them; the depth grows as more filters are added.")

# ---------------- Chapter 22 ----------------
fig, ax = diagram(8, 3.0, ylim=(0, 3.4))
box(ax, 1.0, 1.4, 'x', 0.9, 0.7, fc=TEAL, ec=TEAL, tc='white', bold=True); box(ax, 3.2, 1.4, 'Conv layer\n+ ReLU', 1.6, 0.9); box(ax, 5.4, 1.4, 'Conv layer', 1.6, 0.9)
ax.add_patch(Circle((7.1, 1.4), 0.33, fc=PALE, ec=ORANGE, lw=1.5)); ax.text(7.1, 1.4, '+', ha='center', va='center', fontsize=16, fontweight='bold')
arrow(ax, (1.45, 1.4), (2.4, 1.4)); arrow(ax, (4.0, 1.4), (4.6, 1.4)); arrow(ax, (6.2, 1.4), (6.77, 1.4), 'F(x)')
arrow(ax, (1.0, 1.75), (7.1, 1.73), 'skip connection: x', rad=-0.45, color=ORANGE, toff=(0, 1.05))
arrow(ax, (7.43, 1.4), (8.6, 1.4)); ax.text(9.2, 1.4, 'y = F(x) + x', va='center', fontweight='bold', color=NAVY)
save(fig, 'c22_residual.png')
reg(22, "The fix that changed deep learning", 'c22_residual.png', 'A residual block. The input skips round the two layers and is added back, so the block only has to learn the change F(x).')

L = np.arange(1, 21); plain = 0.12 ** (20 - L + 1); res = np.cumprod(np.r_[1, np.full(19, 0.98)])[::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.3)); ax.semilogy(L, plain, 'o-', c=RED, label='Plain network'); ax.semilogy(L, res, 's-', c=GREEN, label='With skip connections')
ax.set_xlabel('Layer (1 = first layer)'); ax.set_ylabel('Gradient size (log scale)'); ax.legend(frameon=False); ax.set_title('Illustration: gradients reaching each layer of a 20-layer net')
save(fig, 'c22_vanishing.png')
reg(22, "The fix that changed deep learning", 'c22_vanishing.png', 'Illustration of the vanishing gradient. If each layer shrinks the gradient (here by a factor of 0.12), early layers get almost nothing; the skip path keeps it close to 1.')

fig, ax = diagram(8.6, 2.8, ylim=(0, 3.2))
box(ax, 2.6, 2.3, 'Conv layers pre-trained on ImageNet\n(edges, textures, shapes)', 4.2, 0.9, fc=LIGHT, ec=NAVY)
box(ax, 6.9, 2.3, 'Old head\n1000 classes', 1.8, 0.9, fc='#EEEEEE', ec=GRAY, tc=GRAY)
box(ax, 2.6, 0.8, 'Same conv layers (frozen)', 4.2, 0.9, fc=LIGHT, ec=NAVY)
box(ax, 6.9, 0.8, 'New head\nyour 3 classes', 1.8, 0.9, fc=ORANGE, ec=ORANGE, tc='white', bold=True)
arrow(ax, (4.7, 2.3), (6.0, 2.3), color=GRAY); arrow(ax, (4.7, 0.8), (6.0, 0.8)); arrow(ax, (2.6, 1.85), (2.6, 1.25), 'reuse', toff=(0.4, 0))
ax.text(8.95, 0.8, 'train only\nthis part', fontsize=8.5, color=ORANGE, va='center')
save(fig, 'c22_transfer.png')
reg(22, "CNNs do far more than classification", 'c22_transfer.png', 'Transfer learning: keep the feature-extracting layers learned on a big dataset and train only a small new head for your task.')

# ---------------- Chapter 23 ----------------
fig, ax = diagram(9, 3.4, ylim=(0, 4))
for t in range(4):
    x = 1.6 + t * 2.1
    box(ax, x, 0.6, f'x{t + 1}', 0.8, 0.55, fc=LIGHT); box(ax, x, 2.0, f'h{t + 1}', 0.9, 0.7, fc=PALE, ec=ORANGE, bold=True); box(ax, x, 3.4, f'y{t + 1}', 0.8, 0.55, fc='#E3F1EC', ec=TEAL)
    arrow(ax, (x, 0.9), (x, 1.62), 'U', toff=(0.22, 0)); arrow(ax, (x, 2.38), (x, 3.1), 'V', toff=(0.22, 0))
    if t: arrow(ax, (x - 1.65, 2.0), (x - 0.47, 2.0), 'W', toff=(0, 0.2))
arrow(ax, (0.3, 2.0), (1.13, 2.0)); ax.text(0.2, 2.25, 'h0', fontsize=8)
save(fig, 'c23_unrolled.png')
reg(23, "To train and visualize an RNN", 'c23_unrolled.png', 'An RNN unrolled over four time steps. The same weights U, W and V are reused at every step, and the hidden state h carries memory forward.')

fig, ax = diagram(8.6, 3.6, ylim=(0, 4.2))
ax.add_patch(FancyBboxPatch((1.0, 0.7), 7.2, 2.6, boxstyle='round,pad=0.05', fc='#F7F9FC', ec=NAVY, lw=1.5))
ax.plot([0.2, 9.0], [2.9, 2.9], c=ORANGE, lw=3); ax.text(0.2, 3.45, 'cell state s_t (long-term memory)', color=ORANGE, fontsize=9, fontweight='bold')
for x, t, op in [(2.5, 'Forget gate\n(what to erase)', 'x'), (4.6, 'Input gate\n(what to write)', '+'), (6.7, 'Output gate\n(what to show)', '')]:
    box(ax, x, 1.4, t, 1.7, 0.8, fc=TEAL, ec=TEAL, tc='white', fs=8.5)
    if op: ax.add_patch(Circle((x, 2.9), 0.22, fc='white', ec=NAVY, zorder=3)); ax.text(x, 2.9, op, ha='center', va='center', zorder=4, fontweight='bold'); arrow(ax, (x, 1.8), (x, 2.67))
arrow(ax, (6.7, 2.9), (6.7, 1.82), color=GRAY); arrow(ax, (7.55, 1.4), (9.0, 1.4), 'h (output)', toff=(0, 0.25))
ax.text(4.6, 0.3, 'inputs at each step: x_t and previous h', ha='center', fontsize=8.5, color=GRAY)
save(fig, 'c23_lstm.png')
reg(23, "Plain RNNs with sigmoid/tanh", 'c23_lstm.png', 'A simplified LSTM cell. Three gates decide what to erase from, add to, and read out of the cell state, which carries information across many steps.')

# ---------------- Chapter 24 ----------------
fig, ax = diagram(9.6, 3.0, ylim=(0, 3.4))
for i, w in enumerate(['I', 'love', 'tea']):
    box(ax, 0.8 + i * 1.5, 0.6, w, 1.0, 0.5, fc=LIGHT); box(ax, 0.8 + i * 1.5, 1.8, 'enc', 0.9, 0.6, fc=PALE, ec=ORANGE); arrow(ax, (0.8 + i * 1.5, 0.85), (0.8 + i * 1.5, 1.5))
    if i: arrow(ax, (0.8 + (i - 1) * 1.5 + 0.45, 1.8), (0.8 + i * 1.5 - 0.45, 1.8))
box(ax, 5.0, 1.8, 'context\nvector', 1.2, 0.9, fc=ORANGE, ec=ORANGE, tc='white', bold=True); arrow(ax, (4.25, 1.8), (4.4, 1.8))
for i, w in enumerate(['J\'aime', 'le', 'thé']):
    x = 6.6 + i * 1.3; box(ax, x, 1.8, 'dec', 0.9, 0.6, fc='#E3F1EC', ec=TEAL); box(ax, x, 3.0, w, 1.1, 0.5, fc=LIGHT); arrow(ax, (x, 2.1), (x, 2.75))
    arrow(ax, (5.6 if i == 0 else x - 0.85, 1.8), (x - 0.45, 1.8))
ax.text(2.3, 0.05, 'ENCODER reads the input', ha='center', color=ORANGE, fontweight='bold', fontsize=9); ax.text(7.9, 0.9, 'DECODER writes the output', ha='center', color=TEAL, fontweight='bold', fontsize=9)
save(fig, 'c24_seq2seq.png')
reg(24, "For sequence-to-sequence jobs", 'c24_seq2seq.png', 'An encoder-decoder for translation. The encoder squeezes the sentence into a context vector, and the decoder generates the translation word by word.')

E = {'king': (4.0, 4.2), 'queen': (4.4, 1.6), 'man': (1.0, 3.6), 'woman': (1.4, 1.0), 'apple': (7.6, 3.0), 'mango': (8.2, 2.2)}
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for w_, (a_, b_) in E.items(): ax.scatter(a_, b_, c=ORANGE if w_ in ('apple', 'mango') else NAVY, s=50); ax.text(a_ + 0.12, b_ + 0.12, w_, fontsize=10)
for a1, a2 in [('man', 'woman'), ('king', 'queen')]: ax.annotate('', xy=E[a2], xytext=E[a1], arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=1.6))
ax.text(2.4, 2.4, 'same "male -> female"\ndirection', color=GREEN, fontsize=8.5); ax.set_xlabel('dimension 1'); ax.set_ylabel('dimension 2'); ax.set_title('Illustration: words as points in an embedding space')
save(fig, 'c24_embeddings.png')
reg(24, "The prediction-based approach is Word2Vec", 'c24_embeddings.png', 'An illustration of word embeddings. Similar words sit close together, and consistent relationships such as man to woman and king to queen point in the same direction.')
