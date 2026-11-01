from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# ---------------- Chapter 1 ----------------
fig, ax = diagram(8, 3.2)
box(ax, 1.4, 2, 'Human\ninterrogator', 2.0, 1.1, fc=PALE, ec=ORANGE, bold=True)
box(ax, 4.6, 2, 'Typed questions\nand answers\n(screen only)', 2.2, 1.3, fc='white', ec=GRAY)
box(ax, 8.2, 3.0, 'Hidden person', 2.2, 0.8)
box(ax, 8.2, 1.0, 'Hidden machine', 2.2, 0.8, fc='#E3F1EC', ec=TEAL)
arrow(ax, (2.45, 2), (3.45, 2)); arrow(ax, (3.45, 1.75), (2.45, 1.75))
arrow(ax, (5.75, 2.2), (7.05, 2.95)); arrow(ax, (5.75, 1.8), (7.05, 1.05))
ax.text(5, 0.15, 'If the interrogator cannot tell which is which, the machine passes.', ha='center', fontsize=9, color=NAVY)
save(fig, 'c01_turing.png')
reg(1, "There is no single definition of AI", 'c01_turing.png', 'The Turing Test: a person types questions to two hidden partners and must decide which one is the machine.')

fig, ax = diagram(8.4, 3.4, ylim=(0, 4.2))
cols = [(1.6, 'Understanding (NLU)', ['Text classification', 'Sentiment analysis', 'POS tagging', 'Named entity recognition'], LIGHT, NAVY),
        (5.0, 'Both', ['Question answering', 'Summarization', 'Machine translation'], '#E3F1EC', TEAL),
        (8.4, 'Generation (NLG)', ['Image captioning', 'Writing replies', 'Text generation'], PALE, ORANGE)]
for x, title, items, fc, ec in cols:
    box(ax, x, 3.7, title, 2.9, 0.55, fc=ec, ec=ec, tc='white', bold=True)
    for i, it in enumerate(items):
        box(ax, x, 2.95 - i * 0.68, it, 2.7, 0.5, fc=fc, ec=ec)
save(fig, 'c01_nlp_map.png')
reg(1, "Natural Language Processing, or NLP", 'c01_nlp_map.png', 'A map of NLP tasks: some read language, some write it, and some must do both.')

fig, ax = diagram(8.6, 2.2, ylim=(0, 2.4))
steps = ['Question', 'Find useful\ndocuments', 'Pull out\nkey facts', 'List possible\nanswers', 'Write a\nshort reply']
for i, s in enumerate(steps):
    box(ax, 0.9 + i * 2.05, 1.2, s, 1.65, 0.95, fc=ORANGE if i in (0, 4) else LIGHT, ec=ORANGE if i in (0, 4) else NAVY,
        tc='white' if i in (0, 4) else 'black', bold=i in (0, 4))
    if i: arrow(ax, (0.9 + (i - 1) * 2.05 + 0.85, 1.2), (0.9 + i * 2.05 - 0.85, 1.2))
save(fig, 'c01_qa_pipeline.png')
reg(1, "Question answering replies", 'c01_qa_pipeline.png', 'The steps of a question-answering system.')

# ---------------- Chapter 2 ----------------
fig, axs = plt.subplots(1, 3, figsize=(9, 3.1))
def scene(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Rectangle((0, 0), 10, 2.2, color='#C9E4C5')); ax.add_patch(Rectangle((0, 2.2), 10, 5.8, color='#EAF2FB'))
    ax.add_patch(Circle((2.8, 3.3), 1.1, color=ORANGE)); ax.add_patch(Rectangle((5.6, 2.2), 2.8, 3.3, color=NAVY))
for a in axs: scene(a)
axs[0].set_title('Classification'); axs[0].text(5, 7.1, 'Label: "garden scene"', ha='center', fontsize=9, fontweight='bold')
axs[1].set_title('Object detection')
axs[1].add_patch(Rectangle((1.5, 2.0), 2.6, 2.6, fill=False, ec=RED, lw=2)); axs[1].text(1.5, 4.8, 'ball', color=RED, fontsize=9, fontweight='bold')
axs[1].add_patch(Rectangle((5.4, 2.0), 3.2, 3.7, fill=False, ec=RED, lw=2)); axs[1].text(5.4, 5.9, 'box', color=RED, fontsize=9, fontweight='bold')
axs[2].set_title('Segmentation')
axs[2].add_patch(Circle((2.8, 3.3), 1.1, color='#E74C3C', alpha=0.85)); axs[2].add_patch(Rectangle((5.6, 2.2), 2.8, 3.3, color='#8E44AD', alpha=0.85))
axs[2].add_patch(Rectangle((0, 0), 10, 2.2, color='#27AE60', alpha=0.6))
fig.tight_layout(); save(fig, 'c02_image_tasks.png')
reg(2, "Some tasks look inside the picture", 'c02_image_tasks.png', 'One picture, three tasks: classification gives one label, detection draws a box round each object, and segmentation colors every pixel by what it belongs to.')

fig, ax = diagram(8.6, 2.6, ylim=(0, 2.8))
s = ['Speech in\n(Hindi)', 'Speech-to-text', 'Machine\ntranslation', 'Text-to-speech', 'Speech out\n(Telugu)']
for i, t in enumerate(s):
    hl = i in (0, 4)
    box(ax, 0.9 + i * 2.05, 1.5, t, 1.7, 0.95, fc=TEAL if hl else LIGHT, ec=TEAL if hl else NAVY, tc='white' if hl else 'black', bold=hl)
    if i: arrow(ax, (0.9 + (i - 1) * 2.05 + 0.87, 1.5), (0.9 + i * 2.05 - 0.87, 1.5))
ax.text(5, 0.35, 'Speech-to-speech translation chains three systems together.', ha='center', fontsize=9, color=NAVY)
save(fig, 'c02_s2s.png')
reg(2, "Speech is the third big area", 'c02_s2s.png', 'Spoken translation: recognize the words, translate the text, then speak it in the new language.')

rng = np.random.default_rng(0)
normal = np.column_stack([rng.lognormal(3.5, 0.6, 2000), rng.normal(14, 3, 2000) % 24, rng.exponential(5, 2000), rng.poisson(1, 2000)])
fraud = np.column_stack([rng.lognormal(6.5, 0.5, 20), rng.normal(3, 1, 20) % 24, rng.exponential(400, 20), rng.poisson(6, 20)])
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.scatter(normal[:, 0], normal[:, 2] + 0.1, s=6, c=NAVY, alpha=0.35, label='Normal transactions')
ax.scatter(fraud[:, 0], fraud[:, 2] + 0.1, s=40, c=RED, marker='x', label='Fraud')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Amount ($, log scale)'); ax.set_ylabel('Distance from home (km, log scale)')
ax.legend(frameon=False, loc='upper left'); ax.set_title('Frauds stand apart from normal spending')
save(fig, 'c02_fraud.png')
reg(2, "AI is used far beyond pictures", 'c02_fraud.png', 'The data behind Example 3: fraudulent transactions (red) are large and far from home, so an anomaly detector can find them without labels.')

# ---------------- Chapter 3 ----------------
X = np.array([[98, 16.4], [104, 15.9], [110, 19.8], [115, 19.1], [119, 23.0], [123, 22.4], [127, 27.1], [131, 26.3], [134, 31.0], [101, 17.5],
              [152, 52.0], [158, 50.5], [161, 63.2], [165, 57.8], [168, 71.0], [172, 64.5], [175, 80.3], [179, 72.9], [183, 88.0], [188, 81.4]])
y = np.r_[np.zeros(10), np.ones(10)]
years = np.array([1, 3, 4, 6, 7, 9, 11, 13, 16, 20]); sal = np.array([35, 40, 47, 50, 56, 58, 67, 71, 82, 93])
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
clf = LogisticRegression(max_iter=1000).fit(X, y)
xx, yy = np.meshgrid(np.linspace(85, 200, 300), np.linspace(5, 100, 300))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axs[0].contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=['#EAF2FB', PALE])
axs[0].scatter(*X[y == 0].T, c=NAVY, label='Child'); axs[0].scatter(*X[y == 1].T, c=ORANGE, marker='s', label='Adult')
axs[0].set_title('Classification: predict a category'); axs[0].set_xlabel('Height (cm)'); axs[0].set_ylabel('Weight (kg)'); axs[0].legend(frameon=False)
w, b = np.polyfit(years, sal, 1)
axs[1].scatter(years, sal, c=TEAL); xs = np.linspace(0, 21, 2); axs[1].plot(xs, w * xs + b, c=ORANGE, lw=2)
axs[1].set_title('Regression: predict a number'); axs[1].set_xlabel('Years of experience'); axs[1].set_ylabel('Salary ($1000s)')
axs[1].text(1, 88, f'salary = {w:.2f} x years + {b:.2f}', fontsize=9, color=ORANGE)
fig.tight_layout(); save(fig, 'c03_class_vs_reg.png')
reg(3, "Supervised learning has two kinds", 'c03_class_vs_reg.png', 'The two supervised tasks. Left: a classifier splits the space into a child region and an adult region. Right: a regression line predicts salary.')

fig, ax = diagram(8.6, 3.0, ylim=(0, 3.4))
box(ax, 1.3, 2.4, 'Labeled\ntraining data', 1.9, 0.9, fc=TEAL, ec=TEAL, tc='white', bold=True)
box(ax, 4.2, 2.4, 'Learning\nalgorithm', 1.8, 0.9); box(ax, 7.3, 2.4, 'Trained model\ny = f(x)', 2.0, 0.9, fc=PALE, ec=ORANGE, bold=True)
arrow(ax, (2.25, 2.4), (3.3, 2.4)); arrow(ax, (5.1, 2.4), (6.3, 2.4))
box(ax, 4.2, 0.8, 'New, unseen\nexample x', 1.8, 0.8, fc='white', ec=GRAY); box(ax, 9.2, 0.8, 'Prediction', 1.4, 0.7, fc=ORANGE, ec=ORANGE, tc='white', bold=True)
arrow(ax, (5.1, 0.8), (7.3, 1.95), rad=0.25); arrow(ax, (7.6, 1.95), (8.6, 1.1))
ax.text(1.3, 3.25, 'TRAINING PHASE', fontsize=9, color=TEAL, fontweight='bold', ha='center'); ax.text(1.3, 0.8, 'TESTING PHASE', fontsize=9, color=ORANGE, fontweight='bold', ha='center')
save(fig, 'c03_two_phases.png')
reg(3, "Both work in two steps", 'c03_two_phases.png', 'Every supervised model is built in a training phase and judged in a testing phase on data it has never seen.')

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
axs[0].scatter(*X.T, c=GRAY); axs[0].set_title('What the algorithm sees: no labels')
km = KMeans(2, n_init=10, random_state=0).fit(X)
for k, c in zip(range(2), [NAVY, ORANGE]): axs[1].scatter(*X[km.labels_ == k].T, c=c)
axs[1].scatter(*km.cluster_centers_.T, c='black', marker='*', s=220, label='Cluster centers'); axs[1].legend(frameon=False, loc='upper left')
axs[1].set_title('What K-means finds: two groups')
for a in axs: a.set_xlabel('Height (cm)'); a.set_ylabel('Weight (kg)')
fig.tight_layout(); save(fig, 'c03_clustering.png')
reg(3, "In unsupervised learning the data", 'c03_clustering.png', 'Unsupervised learning: with no labels at all, K-means still discovers the two natural groups (children and adults).')

# ---------------- Chapter 4 ----------------
fig, ax = diagram(8.4, 3.2, ylim=(0, 3.6))
box(ax, 1.4, 2.6, 'Small labeled\nset', 1.9, 0.85, fc=TEAL, ec=TEAL, tc='white', bold=True)
box(ax, 4.4, 2.6, 'Train model', 1.7, 0.75); box(ax, 7.6, 2.6, 'Predict labels for\nunlabeled data', 2.3, 0.85)
box(ax, 7.6, 0.8, 'Keep only the\nconfident guesses', 2.3, 0.85, fc=PALE, ec=ORANGE)
box(ax, 4.4, 0.8, 'Add pseudo-labels\nto training set', 2.2, 0.85)
box(ax, 1.4, 0.8, 'Large\nunlabeled pool', 1.9, 0.85, fc='white', ec=GRAY)
arrow(ax, (2.35, 2.6), (3.55, 2.6)); arrow(ax, (5.25, 2.6), (6.45, 2.6)); arrow(ax, (7.6, 2.15), (7.6, 1.25))
arrow(ax, (6.45, 0.8), (5.5, 0.8)); arrow(ax, (4.4, 1.25), (4.4, 2.2), 'retrain', toff=(0.45, 0)); arrow(ax, (2.35, 0.95), (6.45, 2.35), ls='--', color=GRAY, rad=-0.15)
save(fig, 'c04_self_training.png')
reg(4, "Semi-supervised learning trains", 'c04_self_training.png', 'The self-training loop: the model labels the unlabeled data itself, keeps only confident guesses, and retrains.')

labels = [20, 30, 40, 50, 60, 70, 80, 90, 100]
rand = [0.550, 0.707, 0.758, 0.782, 0.820, 0.845, 0.857, 0.870, 0.886]; act = [0.550, 0.731, 0.824, 0.877, 0.883, 0.901, 0.921, 0.925, 0.927]
fig, ax = plt.subplots(figsize=(6.4, 3.5))
ax.plot(labels, rand, 'o-', c=GRAY, label='Random labeling'); ax.plot(labels, act, 's-', c=ORANGE, label='Active learning (uncertainty)')
ax.set_xlabel('Number of labeled images'); ax.set_ylabel('Test accuracy'); ax.legend(frameon=False); ax.set_title('Same labeling budget, better model')
save(fig, 'c04_active_curve.png')
reg(4, "Active learning is a smarter", 'c04_active_curve.png', 'Results from Example 2: asking the expert about the most uncertain images beats labeling at random at almost every budget.')

from sklearn.linear_model import SGDClassifier
rng = np.random.default_rng(0)
def make_batch(t, n=200):
    Xb = rng.normal(0, 1, (n, 2)); yb = (Xb[:, 0] + Xb[:, 1] > 0).astype(int)
    if t >= 10: yb = (Xb[:, 0] - Xb[:, 1] > 0).astype(int)
    return Xb, yb
st = SGDClassifier(loss='log_loss', random_state=0); on = SGDClassifier(loss='log_loss', random_state=0)
X0, y0 = make_batch(0); st.fit(X0, y0); on.partial_fit(X0, y0, classes=[0, 1]); S, O = [], []
for t in range(1, 20):
    Xb, yb = make_batch(t); S.append(st.score(Xb, yb)); O.append(on.score(Xb, yb)); on.partial_fit(Xb, yb)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(range(1, 20), S, 'o-', c=GRAY, label='Static model (trained once)'); ax.plot(range(1, 20), O, 's-', c=TEAL, label='Online model (keeps learning)')
ax.axvline(10, c=RED, ls='--'); ax.text(10.2, 0.6, 'concept drift', color=RED, fontsize=9)
ax.set_xlabel('Batch in the data stream'); ax.set_ylabel('Accuracy'); ax.set_ylim(0.4, 1.02); ax.legend(frameon=False, loc='lower left')
save(fig, 'c04_online.png')
reg(4, "Online learning updates", 'c04_online.png', 'From Example 3: when the rule changes at batch 10, the static model collapses while the online model recovers after one batch.')

fig, ax = diagram(7, 3.0, ylim=(0, 3.4))
box(ax, 2, 1.7, 'Agent', 1.8, 1.0, fc=ORANGE, ec=ORANGE, tc='white', bold=True, fs=11)
box(ax, 8, 1.7, 'Environment', 2.2, 1.0, fc=TEAL, ec=TEAL, tc='white', bold=True, fs=11)
arrow(ax, (2.6, 2.25), (7.4, 2.25), 'action', rad=-0.35, toff=(0, 0.75), fs=10)
arrow(ax, (7.4, 1.15), (2.6, 1.15), 'new state + reward', rad=-0.35, toff=(0, -0.75), fs=10)
save(fig, 'c04_rl_loop.png')
reg(4, "Reinforcement learning (RL) is neither", 'c04_rl_loop.png', 'The reinforcement learning loop: the agent acts, and the environment answers with a new state and a reward.')

# ---------------- Chapter 5 ----------------
rng = np.random.default_rng(42); _age = rng.normal(30, 12, 200); fare = np.round(rng.lognormal(3.0, 0.9, 200), 2)  # same draws as Example 1
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.hist(fare, bins=35, color=LIGHT, ec=NAVY)
ax.axvline(np.median(fare), c=TEAL, lw=2, label=f'median = {np.median(fare):.1f}'); ax.axvline(fare.mean(), c=ORANGE, lw=2, ls='--', label=f'mean = {fare.mean():.1f}')
ax.annotate('long right tail', xy=(150, 3), xytext=(110, 25), arrowprops=dict(arrowstyle='->', color=GRAY), color=GRAY)
ax.set_xlabel('Fare'); ax.set_ylabel('Passengers'); ax.legend(frameon=False); ax.set_title('Right-skewed data: the mean is pulled above the median')
save(fig, 'c05_skew.png')
reg(5, "To profile a numeric column", 'c05_skew.png', "The 'fare' column from Example 1. A few very large values stretch the tail, so the mean sits well above the median.")

rng = np.random.default_rng(7); w = np.r_[rng.normal(70, 6, 99), 250.0]
q1, med, q3 = np.percentile(w, [25, 50, 75]); iqr = q3 - q1; lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
fig, ax = plt.subplots(figsize=(7.4, 2.8))
ax.boxplot(w, orientation='horizontal', widths=0.5, patch_artist=True, boxprops=dict(facecolor=LIGHT, color=NAVY), medianprops=dict(color=ORANGE, lw=2),
           flierprops=dict(marker='o', markerfacecolor=RED, markeredgecolor=RED, markersize=8))
for v, t, tx in [(q1, 'Q1', 95), (med, 'median', 120), (q3, 'Q3', 145)]:
    ax.annotate(t, xy=(v, 1.26), xytext=(tx, 1.42), fontsize=9, color=NAVY, ha='center', arrowprops=dict(arrowstyle='->', color=NAVY, lw=0.8))
ax.axvline(hi, c=RED, ls=':'); ax.text(hi + 2, 0.6, 'upper fence\nQ3 + 1.5 x IQR', fontsize=8, color=RED)
ax.annotate('outlier (typing error?)', xy=(250, 1), xytext=(185, 0.72), arrowprops=dict(arrowstyle='->', color=RED), color=RED, fontsize=9)
ax.set_yticks([]); ax.set_xlabel('Weight (kg)'); ax.set_title('Reading a box plot')
save(fig, 'c05_boxplot.png')
reg(5, "An outlier is a value far away", 'c05_boxplot.png', 'A box plot: the box spans Q1 to Q3, the whiskers reach the furthest normal values, and anything beyond the IQR fences is flagged as an outlier.')

rng = np.random.default_rng(1); n = 600
pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.25, 0.5]); sex = rng.choice(['male', 'female'], n, p=[0.6, 0.4]); age = rng.normal(30, 12, n).clip(1, 75).round()
sibsp = rng.poisson(0.5, n); parch = rng.poisson(0.4, n)
fare = np.round(np.where(pclass == 1, 80, np.where(pclass == 2, 25, 10)) * rng.lognormal(0, 0.4, n), 2)
logit = 1.2 * (sex == 'female') - 0.9 * (pclass - 2) - 0.02 * (age - 30) - 0.4
surv = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
leak = np.where(surv == 1, 1, (rng.random(n) < 0.03).astype(int))
M = np.column_stack([surv, pclass, (sex == 'male').astype(int), age, sibsp + parch + 1, np.log1p(fare), leak])
names = ['survived', 'pclass', 'sex_label', 'age', 'family_size', 'log_fare', 'lifeboat']
C = np.corrcoef(M.T)
fig, ax = plt.subplots(figsize=(6, 4.8))
im = ax.imshow(C, cmap='coolwarm', vmin=-1, vmax=1); ax.set_xticks(range(7)); ax.set_xticklabels(names, rotation=45, ha='right'); ax.set_yticks(range(7)); ax.set_yticklabels(names)
for i in range(7):
    for j in range(7): ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=8)
fig.colorbar(im, fraction=0.046); ax.set_title('Correlation heatmap'); ax.spines[:].set_visible(False)
save(fig, 'c05_heatmap.png')
reg(5, "Advanced EDA moves from tidying", 'c05_heatmap.png', "Correlations for the passenger data in Example 3. 'lifeboat' is almost perfectly correlated with survival, a warning sign of data leakage.")

# ---------------- Chapter 6 ----------------
rng = np.random.default_rng(3); inc = rng.normal(60000, 15000, 80); agev = rng.normal(40, 10, 80)
mm = lambda v: (v - v.min()) / (v.max() - v.min()); zs = lambda v: (v - v.mean()) / v.std()
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
for a, (xv, yv, t) in zip(axs, [(agev, inc, 'Raw values'), (mm(agev), mm(inc), 'Min-Max: both in [0, 1]'), (zs(agev), zs(inc), 'Z-score: mean 0, std 1')]):
    a.scatter(xv, yv, s=12, c=NAVY); a.set_title(t); a.set_xlabel('Age'); a.set_ylabel('Income')
axs[0].ticklabel_format(axis='y', style='plain')
fig.tight_layout(); save(fig, 'c06_scaling.png')
reg(6, "Min-Max normalization squeezes", 'c06_scaling.png', 'The same 80 people before and after scaling. Raw income is in the tens of thousands while age is in tens; after scaling both features are on comparable ranges.')

rng = np.random.default_rng(4); A = rng.normal(70, 10, 100); B = rng.normal(65, 12, 100); T = A + B; Hrs = rng.normal(5, 2, 100)
C = np.corrcoef(np.vstack([A, B, T, Hrs])); nm = ['Marks A', 'Marks B', 'Total marks', 'Sleep hours']
fig, ax = plt.subplots(figsize=(4.8, 4))
im = ax.imshow(C, cmap='coolwarm', vmin=-1, vmax=1); ax.set_xticks(range(4)); ax.set_xticklabels(nm, rotation=30, ha='right'); ax.set_yticks(range(4)); ax.set_yticklabels(nm)
for i in range(4):
    for j in range(4): ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=9)
ax.set_title('Spotting redundant features'); ax.spines[:].set_visible(False); fig.colorbar(im, fraction=0.046)
save(fig, 'c06_redundant.png')
reg(6, "Feature subset selection asks", 'c06_redundant.png', "'Total marks' is built from Marks A and Marks B, so it correlates strongly with both and adds no new information.")
