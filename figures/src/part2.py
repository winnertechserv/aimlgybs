from common import *
from scipy.stats import norm, multivariate_normal as mvn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_circles, load_diabetes
from sklearn.preprocessing import StandardScaler

CHILD = np.array([[98, 16.4], [104, 15.9], [110, 19.8], [115, 19.1], [119, 23.0], [123, 22.4], [127, 27.1], [131, 26.3], [134, 31.0], [101, 17.5]])
ADULT = np.array([[152, 52.0], [158, 50.5], [161, 63.2], [165, 57.8], [168, 71.0], [172, 64.5], [175, 80.3], [179, 72.9], [183, 88.0], [188, 81.4]])
X = np.vstack([CHILD, ADULT]); y = np.r_[np.zeros(10), np.ones(10)]

# ---------------- Chapter 7 ----------------
x = np.array([1, 3, 4, 6, 7, 9, 11, 13, 16, 20.]); s = np.array([35, 40, 47, 50, 56, 58, 67, 71, 82, 93.])
w = np.sum((x - x.mean()) * (s - s.mean())) / np.sum((x - x.mean()) ** 2); w0 = s.mean() - w * x.mean()
fig, ax = plt.subplots(figsize=(6.6, 3.8))
for xi, si in zip(x, s): ax.plot([xi, xi], [si, w * xi + w0], c=RED, lw=1.2)
ax.scatter(x, s, c=NAVY, zorder=3, label='Training data'); xs = np.linspace(0, 21, 2); ax.plot(xs, w * xs + w0, c=ORANGE, lw=2, label=f'Best line: y = {w:.2f}x + {w0:.2f}')
ax.scatter([10], [64], c=TEAL, marker='D', s=60, zorder=3, label='Test person (10 yrs, actual 64)')
ax.plot([], [], c=RED, label='Errors (residuals)'); ax.set_xlabel('Years of experience'); ax.set_ylabel('Salary ($1000s)'); ax.legend(frameon=False, fontsize=8.5)
ax.set_title('Least squares makes the red gaps as small as possible')
save(fig, 'c07_least_squares.png')
reg(7, "'Best' is defined by the method of least squares", 'c07_least_squares.png', 'Least squares on the salary data: the best line minimizes the sum of the squared red gaps.')

D = np.array([[45.3, 1012.7, 29.3], [46.0, 1017.9, 29.3], [53.7, 1021.4, 25.2], [62.7, 1023.2, 22.0], [81.8, 1023.1, 18.8], [59.8, 1021.6, 23.4],
              [57.2, 1018.5, 23.7], [82.4, 1022.0, 18.6], [81.5, 1028.3, 18.7], [43.5, 1011.0, 32.0], [48.4, 1024.5, 27.1], [68.5, 1027.9, 21.3],
              [55.5, 1014.5, 24.1], [42.9, 1016.8, 31.6], [42.8, 1012.3, 30.6]])
A = np.column_stack([np.ones(15), D[:, :2]]); wv = np.linalg.pinv(A) @ D[:, 2]
fig = plt.figure(figsize=(6.4, 4.6)); ax = fig.add_subplot(projection='3d')
hh, pp = np.meshgrid(np.linspace(40, 85, 10), np.linspace(1010, 1030, 10))
ax.plot_surface(hh, pp, wv[0] + wv[1] * hh + wv[2] * pp, alpha=0.35, color=ORANGE)
ax.scatter(D[:, 0], D[:, 1], D[:, 2], c=NAVY, s=30, depthshade=False)
ax.set_xlabel('Humidity (%)'); ax.set_ylabel('Pressure (hPa)'); ax.set_zlabel('Temp (C)'); ax.set_title('Two inputs: the model is a flat plane')
ax.view_init(22, -55); save(fig, 'c07_plane.png')
reg(7, "Multiple linear regression uses two or more inputs", 'c07_plane.png', 'Multiple linear regression on the weather data from Example 2: with two inputs, the fitted model is a flat plane.')

Xd, yd = load_diabetes(return_X_y=True); Xd = StandardScaler().fit_transform(Xd[:309]); yd = yd[:309]
Ad = np.column_stack([np.ones(len(Xd)), Xd]); wg = np.zeros(Ad.shape[1]); hist = []
for e in range(300):
    wg -= 0.1 * Ad.T @ (Ad @ wg - yd) / len(yd); hist.append(np.mean((Ad @ wg - yd) ** 2))
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.plot(range(1, 301), hist, c=NAVY, lw=2); ax.axhline(np.mean((Ad @ np.linalg.pinv(Ad) @ yd - yd) ** 2), c=ORANGE, ls='--', label='Closed-form answer')
ax.set_yscale('log'); ax.set_xlabel('Gradient descent step'); ax.set_ylabel('Training MSE (log scale)'); ax.legend(frameon=False); ax.set_title('Gradient descent walks down to the closed-form answer')
save(fig, 'c07_gd.png')
reg(7, "To judge a regression model", 'c07_gd.png', 'Gradient descent on the diabetes data (Example 3): the error falls step by step until it reaches the exact least-squares solution.')

# ---------------- Chapter 8 ----------------
temp, hum = D[:, 2], D[:, 0]
tt = np.array([19.5, 22.8, 26.0, 28.4, 31.2]); ht = np.array([77.5, 61.4, 50.5, 44.8, 42.8])
fig, ax = plt.subplots(figsize=(6.6, 3.8)); xs = np.linspace(18, 32.5, 200)
ax.scatter(temp, hum, c=NAVY, label='Training days'); ax.scatter(tt, ht, c=RED, marker='x', s=60, label='New test days')
for M, c, ls in [(1, GRAY, '--'), (2, ORANGE, '-'), (3, TEAL, ':')]:
    wM = np.linalg.pinv(np.vander(temp, M + 1, increasing=True)) @ hum
    ax.plot(xs, np.vander(xs, M + 1, increasing=True) @ wM, c=c, ls=ls, lw=2, label=f'M = {M}')
ax.set_xlabel('Temperature (C)'); ax.set_ylabel('Humidity (%)'); ax.legend(frameon=False); ax.set_title('A straight line misses the bend; M = 2 follows it')
save(fig, 'c08_poly_fits.png')
reg(8, "To do that, build the design matrix", 'c08_poly_fits.png', 'Example 1: polynomials of degree 1, 2 and 3 fitted to humidity vs temperature, with five new test days.')

rng = np.random.default_rng(0); f = lambda v: np.sin(2 * np.pi * v)
x_te = rng.uniform(0, 1, 500); y_te = f(x_te) + rng.normal(0, 0.3, 500)   # same draws as Example 2
x10 = np.linspace(0, 1, 10); y10 = f(x10) + rng.normal(0, 0.3, 10); xs = np.linspace(0, 1, 300)
fig, axs = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
for a, M in zip(axs, [0, 1, 3, 9]):
    wM = np.linalg.lstsq(np.vander(x10, M + 1, increasing=True), y10, rcond=None)[0]
    a.plot(xs, f(xs), c=GREEN, lw=1.5, alpha=0.7); a.plot(xs, np.vander(xs, M + 1, increasing=True) @ wM, c=ORANGE, lw=2)
    a.scatter(x10, y10, c=NAVY, zorder=3); a.set_ylim(-2.2, 2.2); a.set_title(f'M = {M}'); a.set_xlabel('x')
axs[0].set_ylabel('y'); axs[0].plot([], [], c=GREEN, label='true curve'); axs[0].plot([], [], c=ORANGE, label='fitted'); axs[0].legend(frameon=False, fontsize=8, loc='lower left')
fig.suptitle('Underfitting (M = 0, 1), a good fit (M = 3) and overfitting (M = 9)', color=NAVY, fontweight='bold'); fig.tight_layout()
save(fig, 'c08_overfit_panels.png')
reg(8, "How do you choose M?", 'c08_overfit_panels.png', 'Ten noisy points from a sine wave. M = 9 passes through every point but swings wildly between them: that is overfitting.')

tr, te = [], []
for M in range(10):
    wM = np.linalg.lstsq(np.vander(x10, M + 1, increasing=True), y10, rcond=None)[0]
    tr.append(np.sqrt(np.mean((np.vander(x10, M + 1, increasing=True) @ wM - y10) ** 2))); te.append(np.sqrt(np.mean((np.vander(x_te, M + 1, increasing=True) @ wM - y_te) ** 2)))
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(range(10), tr, 'o-', c=NAVY, label='Training error'); ax.plot(range(10), te, 's-', c=RED, label='Test error')
ax.set_xlabel('Polynomial degree M'); ax.set_ylabel('RMSE'); ax.legend(frameon=False); ax.set_title('Training error keeps falling; test error turns back up')
save(fig, 'c08_rmse_vs_m.png')
reg(8, "How do you choose M?", 'c08_rmse_vs_m.png', 'Choosing M: always judge on test (or validation) data, because training error falls no matter what.')

rng = np.random.default_rng(2); xr = np.linspace(0, 1, 10); yr = f(xr) + rng.normal(0, 0.3, 10); Phi = np.vander(xr, 10, increasing=True)
fig, axs = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
for a, lam in zip(axs, [0, 1e-4, 1]):
    wl = np.linalg.lstsq(Phi, yr, rcond=None)[0] if lam == 0 else np.linalg.solve(Phi.T @ Phi + lam * np.eye(10), Phi.T @ yr)
    a.plot(xs, f(xs), c=GREEN, alpha=0.7); a.plot(xs, np.vander(xs, 10, increasing=True) @ wl, c=ORANGE, lw=2); a.scatter(xr, yr, c=NAVY, zorder=3)
    a.set_ylim(-1.8, 1.8); a.set_title(f'lambda = {lam:g}' + ('  (overfits)' if lam == 0 else '  (good)' if lam < 0.01 else '  (underfits)'))
fig.tight_layout(); save(fig, 'c08_ridge.png')
reg(8, "The first fix is more data", 'c08_ridge.png', 'Ridge regression on the same degree-9 model (Example 3): a small penalty tames the wiggles, a large one flattens the curve.')

# ---------------- Chapter 9 ----------------
t = np.array([140, 46.0]); d = np.sqrt(((X - t) ** 2).sum(1)); order = np.argsort(d)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.scatter(*CHILD.T, c=NAVY, label='Child'); ax.scatter(*ADULT.T, c=ORANGE, marker='s', label='Adult')
ax.scatter(*t, c=RED, marker='*', s=250, zorder=4, label='Test person (140 cm, 46 kg)')
for K, ls in [(1, '-'), (3, '--'), (5, ':')]:
    r = d[order[K - 1]]; ax.add_patch(Circle(t, r, fill=False, ec=GRAY, ls=ls, lw=1.4)); ax.text(t[0] + r * 0.72, t[1] + r * 0.72, f'K={K}', color=GRAY, fontsize=9)
ax.set_aspect('equal'); ax.set_xlim(90, 195); ax.set_ylim(5, 95); ax.set_xlabel('Height (cm)'); ax.set_ylabel('Weight (kg)'); ax.legend(frameon=False, loc='upper left', fontsize=8.5)
ax.set_title('K = 1 and 3 vote Adult; K = 5 votes Child')
save(fig, 'c09_knn.png')
reg(9, "The nearest-neighbor method", 'c09_knn.png', 'K-NN for the second test person in Example 1. Each circle holds the K nearest neighbors; changing K changes the vote.')

fig, ax = diagram(7.4, 3.6, xlim=(0, 10), ylim=(0, 5))
ax.text(6.2, 4.7, 'Actual class', ha='center', fontsize=10, fontweight='bold', color=NAVY); ax.text(0.4, 2.3, 'Predicted', rotation=90, va='center', fontsize=10, fontweight='bold', color=NAVY)
ax.text(4.8, 4.15, 'Positive', ha='center'); ax.text(7.6, 4.15, 'Negative', ha='center'); ax.text(2.2, 3.1, 'Positive', ha='center'); ax.text(2.2, 1.5, 'Negative', ha='center')
cells = [(4.8, 3.1, 'TP = 60', '#D4EDDA'), (7.6, 3.1, 'FP = 6\n(false alarm)', '#F8D7DA'), (4.8, 1.5, 'FN = 4\n(missed)', '#F8D7DA'), (7.6, 1.5, 'TN = 101', '#D4EDDA')]
for cx, cy, tx, fc in cells: box(ax, cx, cy, tx, 2.6, 1.45, fc=fc, ec=GRAY, fs=10, bold=True, round_=0.02)
ax.text(5, 0.25, 'Precision = TP/(TP+FP) = 0.909      Recall = TP/(TP+FN) = 0.938', ha='center', fontsize=9.5, color=NAVY)
save(fig, 'c09_confusion.png')
reg(9, "Accuracy alone can mislead", 'c09_confusion.png', 'The confusion matrix from Example 2 (cancer screening). Green cells are correct; red cells are the two kinds of mistake.')

def ellipse(ax, mu, S, c):
    vals, vecs = np.linalg.eigh(S); ang = np.linspace(0, 2 * np.pi, 200)
    for k in (1, 2, 3):
        pts = mu[:, None] + vecs @ np.diag(k * np.sqrt(vals)) @ np.vstack([np.cos(ang), np.sin(ang)]); ax.plot(*pts, c=c, lw=1, alpha=0.7)
mc, ma = CHILD.mean(0), ADULT.mean(0)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.scatter(*CHILD.T, c=NAVY, label='Child'); ax.scatter(*ADULT.T, c=ORANGE, marker='s', label='Adult')
ellipse(ax, mc, np.cov(CHILD, rowvar=False), NAVY); ellipse(ax, ma, np.cov(ADULT, rowvar=False), ORANGE)
x2 = np.array([137, 38.]); ax.scatter(*x2, c=RED, marker='*', s=250, zorder=4, label='x2 = (137 cm, 38 kg)')
ax.plot(*np.vstack([x2, mc]).T, c=NAVY, ls='--'); ax.plot(*np.vstack([x2, ma]).T, c=ORANGE, ls='--')
ax.set_xlabel('Height (cm)'); ax.set_ylabel('Weight (kg)'); ax.legend(frameon=False, loc='upper left', fontsize=8.5); ax.set_xlim(80, 205); ax.set_ylim(-5, 110)
ax.set_title('Rings show 1, 2 and 3 Mahalanobis units from each mean')
save(fig, 'c09_mahalanobis.png')
reg(9, "Euclidean distance ignores", 'c09_mahalanobis.png', 'Why Mahalanobis distance disagrees with plain distance for x2: x2 lies just inside the third adult ring, but well outside the narrow child rings.')

# ---------------- Chapter 10 ----------------
hC, hA = CHILD[:, 0], ADULT[:, 0]; xs = np.linspace(80, 210, 400)
pc, pa = norm(hC.mean(), hC.std(ddof=1)).pdf(xs), norm(hA.mean(), hA.std(ddof=1)).pdf(xs)
fig, axs = plt.subplots(2, 1, figsize=(6.6, 4.6), sharex=True)
axs[0].plot(xs, pc, c=NAVY, lw=2, label='p(height | Child)'); axs[0].plot(xs, pa, c=ORANGE, lw=2, label='p(height | Adult)'); axs[0].legend(frameon=False, fontsize=8.5); axs[0].set_title('Likelihoods (one Gaussian per class)')
axs[1].plot(xs, pc / (pc + pa), c=NAVY, lw=2, label='P(Child | height)'); axs[1].plot(xs, pa / (pc + pa), c=ORANGE, lw=2, label='P(Adult | height)'); axs[1].set_title('Posteriors (equal priors)')
axs[1].legend(frameon=False, fontsize=8.5); axs[1].set_xlabel('Height (cm)'); fig.tight_layout()
save(fig, 'c10_bayes_1d.png')
reg(10, "Instead of distances, the Bayes classifier", 'c10_bayes_1d.png', 'Bayes classifier using height alone. Top: how well each height fits each class. Bottom: Bayes rule turns these into class probabilities that add up to 1.')

fig, ax = plt.subplots(figsize=(6.6, 3.2))
for pr, ls, lab in [(0.5, '-', 'equal priors (0.5 / 0.5)'), (0.9, '--', 'school priors (0.9 child)')]:
    post = pr * pc / (pr * pc + (1 - pr) * pa); ax.plot(xs, post, c=NAVY, ls=ls, lw=2, label=lab)
    ax.axvline(xs[np.argmin(abs(post - 0.5))], c=GRAY, ls=ls, lw=1)
ax.set_xlabel('Height (cm)'); ax.set_ylabel('P(Child | height)'); ax.legend(frameon=False); ax.set_title('A stronger prior moves the decision boundary')
save(fig, 'c10_priors.png')
reg(10, "Priors matter", 'c10_priors.png', 'When children are much more common, the boundary (gray lines, where the probability is 0.5) moves to the right, so more borderline people are called children.')

rng = np.random.default_rng(0)
Ad = np.vstack([rng.normal([0, 0], 0.6, (150, 2)), rng.normal([4, 4], 0.6, (150, 2))]); Bd = rng.normal([2, 2], 0.6, (300, 2))
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.8))
gg = np.meshgrid(np.linspace(-2, 6, 200), np.linspace(-2, 6, 200)); P = np.c_[gg[0].ravel(), gg[1].ravel()]
for a, K in zip(axs, (1, 2)):
    g = GaussianMixture(K, random_state=0).fit(Ad); Z = np.exp(g.score_samples(P)).reshape(gg[0].shape)
    a.scatter(*Ad.T, s=8, c=NAVY, alpha=0.6, label='Class A'); a.scatter(*Bd.T, s=8, c=ORANGE, alpha=0.6, label='Class B')
    a.contour(*gg, Z, levels=6, colors=NAVY, linewidths=1); a.set_title(f'Class A modeled with K = {K} ' + ('Gaussian' if K == 1 else 'Gaussians (GMM)'))
    a.set_aspect('equal')
axs[0].legend(frameon=False, loc='upper left', fontsize=8.5); fig.tight_layout()
save(fig, 'c10_gmm.png')
reg(10, "Real classes are often not a single bell shape", 'c10_gmm.png', 'Example 3: one Gaussian cannot describe class A, which has two clumps; its single wide bell even covers class B. A two-component GMM fits each clump.')

# ---------------- Chapter 11 ----------------
fig, ax = plt.subplots(figsize=(4.8, 4.2)); g1 = np.linspace(-0.5, 1.5, 300); G = np.meshgrid(g1, g1)
ax.contourf(*G, G[0] + G[1] - 1, levels=[-5, 0, 5], colors=['#EAF2FB', PALE]); ax.plot(g1, 1 - g1, c=NAVY, lw=2)
for p, lab in [((1, 0), '(1,0): g = 0'), ((0, 1), '(0,1): g = 0'), ((0, 0), '(0,0): g = -1'), ((1, 1), '(1,1): g = +1')]:
    ax.scatter(*p, c=RED, zorder=3); ax.text(p[0] + 0.05, p[1] + 0.06, lab, fontsize=9)
ax.text(-0.4, -0.35, 'negative side', color=NAVY); ax.text(0.9, 1.35, 'positive side', color=ORANGE); ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_aspect('equal')
ax.set_title('g(x) = x1 + x2 - 1')
save(fig, 'c11_line_sides.png')
reg(11, "A straight boundary is described", 'c11_line_sides.png', 'The sign of g(x) tells you which side of the line a point is on.')

def bayes_boundary(ax, m0, S0, m1, S1, title):
    gg = np.meshgrid(np.linspace(-4, 7, 300), np.linspace(-4, 7, 300)); P = np.dstack(gg)
    Z = mvn(m0, S0).logpdf(P) - mvn(m1, S1).logpdf(P)
    rng = np.random.default_rng(1); a = rng.multivariate_normal(m0, S0, 120); b = rng.multivariate_normal(m1, S1, 120)
    ax.scatter(*a.T, s=8, c=NAVY, alpha=0.6); ax.scatter(*b.T, s=8, c=ORANGE, alpha=0.6); ax.contour(*gg, Z, levels=[0], colors=RED, linewidths=2)
    ax.plot(*np.vstack([m0, m1]).T, 'k:', lw=1); ax.set_title(title); ax.set_aspect('equal'); ax.set_xlim(-4, 7); ax.set_ylim(-4, 7)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.7))
bayes_boundary(axs[0], [0, 0], np.eye(2), [3, 3], np.eye(2), 'Case 1: same, round covariance\nstraight, cuts the gap in half')
bayes_boundary(axs[1], [0, 0], [[2, 1.5], [1.5, 2]], [3, 3], [[2, 1.5], [1.5, 2]], 'Case 2: same covariance\nstraight but tilted')
bayes_boundary(axs[2], [0, 0], [[0.4, 0], [0, 0.4]], [2.5, 2.5], [[4, 0], [0, 4]], 'Case 3: different covariances\ncurved boundary')
fig.tight_layout(); save(fig, 'c11_bayes_cases.png')
reg(11, "For a Gaussian Bayes classifier", 'c11_bayes_cases.png', 'The three covariance cases. The red line is the Bayes decision boundary; the dotted line joins the two class means.')

rng = np.random.default_rng(1)
Xp = np.vstack([rng.normal([1, 1], 0.5, (20, 2)), rng.normal([-1, -1], 0.5, (20, 2))]); yp = np.r_[np.ones(20), -np.ones(20)]
Xh = np.column_stack([np.ones(40), Xp]); wp = np.array([2.0, -1.0, 0.5]); lines = [wp.copy()]
for epoch in range(5):
    for xn, yn in zip(Xh, yp):
        s_ = wp @ xn
        if yn == 1 and s_ <= 0: wp = wp + 0.1 * xn; lines.append(wp.copy())
        elif yn == -1 and s_ > 0: wp = wp - 0.1 * xn; lines.append(wp.copy())
fig, ax = plt.subplots(figsize=(5.4, 4.4)); xs = np.linspace(-2.5, 2.5, 2)
ax.scatter(*Xp[yp == 1].T, c=ORANGE, marker='s', label='+1'); ax.scatter(*Xp[yp == -1].T, c=NAVY, label='-1')
for i, L in enumerate(lines):
    c, lw, a_ = (RED, 1.5, 1) if i == 0 else (GREEN, 2.5, 1) if i == len(lines) - 1 else (GRAY, 0.8, 0.5)
    ax.plot(xs, -(L[0] + L[1] * xs) / L[2], c=c, lw=lw, alpha=a_)
ax.plot([], [], c=RED, label='starting line'); ax.plot([], [], c=GRAY, label='after each mistake'); ax.plot([], [], c=GREEN, lw=2.5, label='final line')
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.legend(frameon=False, fontsize=8, loc='upper left'); ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_title('Perceptron: each mistake nudges the line')
save(fig, 'c11_perceptron.png')
reg(11, "The perceptron learns a linear", 'c11_perceptron.png', f'The perceptron from Example 2 starting from a deliberately bad line. Every gray line is one update ({len(lines) - 1} in total) until no point is misclassified.')

rng = np.random.default_rng(0); cen = [([0, 0], -1), ([1, 1], -1), ([0, 1], 1), ([1, 0], 1)]
Xx = np.vstack([rng.normal(c, 0.1, (25, 2)) for c, _ in cen]); yx = np.repeat([l for _, l in cen], 25)
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.8))
for a in axs: a.scatter(*Xx[yx == 1].T, c=ORANGE, marker='s'); a.scatter(*Xx[yx == -1].T, c=NAVY); a.set_aspect('equal'); a.set_xlim(-0.4, 1.4); a.set_ylim(-0.4, 1.4)
for off in (-0.6, 0.0, 0.6): axs[0].plot([-0.4, 1.4], [1.4 + off - 0.0, -0.4 + off], c=RED, ls='--', alpha=0.6)
axs[0].set_title('XOR: no straight line can separate it')
G = np.meshgrid(np.linspace(-0.4, 1.4, 200), np.linspace(-0.4, 1.4, 200)); Z = KNeighborsClassifier(5).fit(Xx, yx).predict(np.c_[G[0].ravel(), G[1].ravel()]).reshape(G[0].shape)
axs[1].contourf(*G, Z, levels=[-2, 0, 2], colors=['#EAF2FB', PALE]); axs[1].set_title('K-NN learns a curved (nonlinear) boundary'); fig.tight_layout()
save(fig, 'c11_xor.png')
reg(11, "The perceptron convergence theorem", 'c11_xor.png', 'The XOR pattern from Example 3. Every straight line leaves points on the wrong side, but a nonlinear method separates the classes easily.')

# ---------------- Chapter 12 ----------------
svm = SVC(kernel='linear', C=1e6).fit(Xp, yp); w_, b_ = svm.coef_[0], svm.intercept_[0]
fig, ax = plt.subplots(figsize=(5.6, 4.6)); xs = np.linspace(-2.5, 2.5, 2)
ax.scatter(*Xp[yp == 1].T, c=ORANGE, marker='s', label='+1'); ax.scatter(*Xp[yp == -1].T, c=NAVY, label='-1')
for off, ls in [(0, '-'), (1, '--'), (-1, '--')]: ax.plot(xs, (off - b_ - w_[0] * xs) / w_[1], c=GREEN, ls=ls, lw=2 if off == 0 else 1.2)
ax.fill_between(xs, (1 - b_ - w_[0] * xs) / w_[1], (-1 - b_ - w_[0] * xs) / w_[1], color=GREEN, alpha=0.12)
ax.scatter(*svm.support_vectors_.T, s=220, facecolors='none', edgecolors=RED, lw=2, label='support vectors')
ax.plot(xs, -(wp[0] + wp[1] * xs) / wp[2], c=GRAY, ls=':', lw=1.5, label='perceptron line')
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.legend(frameon=False, fontsize=8, loc='upper left'); ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title(f'Margin = 1/||w|| = {1 / np.linalg.norm(w_):.3f}')
save(fig, 'c12_max_margin.png')
reg(12, "For any point x, its distance", 'c12_max_margin.png', 'The maximum-margin hyperplane from Example 1. The shaded street is as wide as possible; only the two circled support vectors touch it.')

rng = np.random.default_rng(0); Xo = np.vstack([rng.normal([0, 0], 1.0, (150, 2)), rng.normal([2, 2], 1.0, (150, 2))]); yo = np.r_[-np.ones(150), np.ones(150)]
fig, axs = plt.subplots(1, 2, figsize=(9.4, 4)); xs = np.linspace(-3.5, 5.5, 2)
for a, C in zip(axs, (0.01, 100)):
    m = SVC(kernel='linear', C=C).fit(Xo, yo); w_, b_ = m.coef_[0], m.intercept_[0]
    a.scatter(*Xo[yo == 1].T, s=10, c=ORANGE); a.scatter(*Xo[yo == -1].T, s=10, c=NAVY)
    for off, ls in [(0, '-'), (1, '--'), (-1, '--')]: a.plot(xs, (off - b_ - w_[0] * xs) / w_[1], c=GREEN, ls=ls)
    a.scatter(*m.support_vectors_.T, s=40, facecolors='none', edgecolors=RED, lw=0.8)
    a.set_xlim(-3.5, 5.5); a.set_ylim(-3.5, 5.5); a.set_aspect('equal'); a.set_title(f'C = {C}: margin {1 / np.linalg.norm(w_):.2f}, {len(m.support_)} support vectors')
fig.tight_layout(); save(fig, 'c12_soft_margin.png')
reg(12, "Case 2 is data that overlaps", 'c12_soft_margin.png', 'Soft margin: a small C gives a wide street with many support vectors (circled); a large C squeezes the street to make fewer training errors.')

Xc, yc = make_circles(n_samples=300, factor=0.4, noise=0.08, random_state=0)
fig = plt.figure(figsize=(10, 3.8)); a1 = fig.add_subplot(1, 2, 1); a2 = fig.add_subplot(1, 2, 2, projection='3d')
a1.scatter(*Xc[yc == 0].T, s=10, c=NAVY); a1.scatter(*Xc[yc == 1].T, s=10, c=ORANGE); a1.set_aspect('equal'); a1.set_title('Input space: no straight line works')
z = (Xc ** 2).sum(1); a2.scatter(Xc[:, 0], Xc[:, 1], z, c=np.where(yc == 1, ORANGE, NAVY), s=8, depthshade=False)
gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 5), np.linspace(-1.2, 1.2, 5)); a2.plot_surface(gx, gy, np.full_like(gx, 0.5), alpha=0.25, color=GREEN)
a2.set_title('Add z = x1² + x2²: a flat plane separates them'); a2.view_init(15, 30)
save(fig, 'c12_kernel_lift.png')
reg(12, "Case 3 is data that needs a curved boundary", 'c12_kernel_lift.png', 'The idea behind kernels: rings that no line can split in 2-D become separable by a flat plane once a third feature is added.')

fig, axs = plt.subplots(1, 2, figsize=(9, 3.8)); G = np.meshgrid(np.linspace(-1.4, 1.4, 250), np.linspace(-1.4, 1.4, 250)); P = np.c_[G[0].ravel(), G[1].ravel()]
for a, (name, m) in zip(axs, [('Linear kernel', SVC(kernel='linear')), ('Gaussian (RBF) kernel', SVC(kernel='rbf', gamma=1))]):
    m.fit(Xc, yc); Z = m.predict(P).reshape(G[0].shape); a.contourf(*G, Z, levels=[-0.5, 0.5, 1.5], colors=['#EAF2FB', PALE])
    a.scatter(*Xc[yc == 0].T, s=8, c=NAVY); a.scatter(*Xc[yc == 1].T, s=8, c=ORANGE); a.set_aspect('equal'); a.set_title(f'{name}: accuracy {m.score(Xc, yc):.2f}')
fig.tight_layout(); save(fig, 'c12_rbf.png')
reg(12, "Common kernels are", 'c12_rbf.png', 'The same rings classified by a linear SVM and by an RBF-kernel SVM. The kernel lets the SVM draw a circular boundary.')
