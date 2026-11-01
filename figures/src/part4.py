from common import *
from sklearn.datasets import load_sample_image
from scipy.stats import norm

# ---------------- Chapter 25 ----------------
fig, ax = diagram(8.6, 3.0, ylim=(0, 3.4))
box(ax, 0.8, 2.6, 'Q', 0.8, 0.6, fc=ORANGE, ec=ORANGE, tc='white', bold=True); box(ax, 0.8, 1.6, 'K', 0.8, 0.6, fc=TEAL, ec=TEAL, tc='white', bold=True)
box(ax, 0.8, 0.5, 'V', 0.8, 0.6, fc=NAVY, ec=NAVY, tc='white', bold=True)
for i, t in enumerate(['dot products\nq . k', 'divide by\nsqrt(d)', 'softmax\n(weights sum to 1)']):
    box(ax, 2.7 + i * 2.0, 2.1, t, 1.8, 0.85, fs=8.5)
    if i: arrow(ax, (2.7 + (i - 1) * 2.0 + 0.9, 2.1), (2.7 + i * 2.0 - 0.9, 2.1))
arrow(ax, (1.2, 2.6), (1.9, 2.2)); arrow(ax, (1.2, 1.6), (1.9, 2.0))
box(ax, 8.3, 1.3, 'weighted sum\nof values\n= context', 1.6, 1.0, fc=PALE, ec=ORANGE, bold=True)
arrow(ax, (7.5, 2.1), (8.1, 1.82)); arrow(ax, (1.2, 0.5), (7.5, 1.05), rad=0.12)
save(fig, 'c25_sdpa.png')
reg(25, "The terminology of SDPA", 'c25_sdpa.png', 'Scaled dot-product attention: score each key against the query, scale, turn the scores into weights with softmax, then take the weighted sum of the values.')

words = ['The', 'animal', 'did', 'not', 'cross', 'the', 'road', 'because', 'it', 'was', 'tired']
rng = np.random.default_rng(2); S = rng.normal(0, 0.6, (11, 11)); np.fill_diagonal(S, 1.5)
S[8, 1] = 4.0; S[8, 6] = 1.8; S[10, 1] = 2.2; S[10, 8] = 2.5; S[4, 6] = 2.4; S[4, 1] = 2.0
Wt = np.exp(S) / np.exp(S).sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.2, 5.2)); ax.imshow(Wt, cmap='Oranges')
ax.set_xticks(range(11)); ax.set_xticklabels(words, rotation=60, ha='right'); ax.set_yticks(range(11)); ax.set_yticklabels(words)
ax.set_xlabel('attends to (keys)'); ax.set_ylabel('word being encoded (query)'); ax.set_title("Illustration: self-attention weights\n('it' looks mostly at 'animal')")
ax.add_patch(Rectangle((0.5, 7.5), 1, 1, fill=False, ec=NAVY, lw=2)); ax.spines[:].set_visible(False)
save(fig, 'c25_attention_map.png')
reg(25, "For self-attention", 'c25_attention_map.png', "An illustration of self-attention weights. Each row shows how much one word draws on every other word; the word 'it' puts most of its weight on 'animal'.")

fig, ax = diagram(5.6, 4.6, xlim=(0, 6), ylim=(0, 6))
ax.add_patch(FancyBboxPatch((1.0, 0.9), 4.0, 4.4, boxstyle='round,pad=0.05', fc='#F7F9FC', ec=NAVY, lw=1.5)); ax.text(5.2, 5.0, 'x N\nlayers', color=NAVY, fontsize=9)
for yy, t, fc in [(1.6, 'Multi-head\nself-attention', PALE), (2.65, 'Add & Norm', LIGHT), (3.7, 'Position-wise\nfeedforward', '#E3F1EC'), (4.7, 'Add & Norm', LIGHT)]:
    box(ax, 3.0, yy, t, 2.6, 0.75 if '\n' in t else 0.55, fc=fc, ec=NAVY, fs=9)
for y1, y2 in [(0.3, 1.2), (2.0, 2.37), (2.92, 3.32), (4.08, 4.42), (4.98, 5.7)]: arrow(ax, (3.0, y1), (3.0, y2))
arrow(ax, (3.0, 0.95), (1.7, 2.65), rad=-0.5, color=ORANGE, ls='--'); arrow(ax, (3.0, 3.0), (1.7, 4.7), rad=-0.5, color=ORANGE, ls='--')
ax.text(3.0, 0.1, 'input embeddings + position encoding', ha='center', fontsize=8.5); ax.text(0.15, 3.0, 'residual\nconnections', color=ORANGE, fontsize=8.5)
save(fig, 'c25_encoder_layer.png')
reg(25, "An encoder layer contains two sub-layers", 'c25_encoder_layer.png', 'One Transformer encoder layer: self-attention then a feedforward network, each wrapped in a residual connection and layer normalisation. The layer is stacked N times.')

# ---------------- Chapter 26 ----------------
d, T = 64, 50; PE = np.zeros((T, d)); pos = np.arange(T)[:, None]; i2 = np.arange(0, d, 2)
PE[:, 0::2] = np.sin(pos / 10000 ** (i2 / d)); PE[:, 1::2] = np.cos(pos / 10000 ** (i2 / d))
fig, ax = plt.subplots(figsize=(7, 3.6)); im = ax.imshow(PE, cmap='RdBu', aspect='auto'); fig.colorbar(im, fraction=0.03)
ax.set_xlabel('Embedding dimension'); ax.set_ylabel('Position in sentence'); ax.set_title('Sinusoidal position encoding (d = 64)')
save(fig, 'c26_posenc.png')
reg(26, "Each position t gets its own vector", 'c26_posenc.png', 'Every row is the position vector for one word position. Early dimensions change quickly from row to row, later ones slowly, so every position gets a unique pattern.')

N = np.arange(10, 4001); dm = 512
fig, ax = plt.subplots(figsize=(6.4, 3.4)); ax.loglog(N, N ** 2 * dm, c=ORANGE, lw=2, label='Self-attention: N² x d'); ax.loglog(N, N * dm ** 2, c=NAVY, lw=2, label='Recurrent layer: N x d²')
ax.axvline(dm, c=GREY, ls='--'); ax.text(dm * 1.1, 1e7, 'N = d = 512', color=GREY, fontsize=9)
ax.set_xlabel('Sequence length N (log scale)'); ax.set_ylabel('Operations per layer (log scale)'); ax.legend(frameon=False); ax.set_title('Attention cost grows with the square of length')
save(fig, 'c26_complexity.png')
reg(26, "Two engineering details complete the Transformer", 'c26_complexity.png', 'Cost per layer with d = 512. Self-attention is cheaper than a recurrent layer for sentences shorter than d, but its cost grows with the square of the length.')

fig, axs = plt.subplots(1, 2, figsize=(8, 3.6)); toks = ['I', 'love', '[MASK]', 'tea']
for ax_, M, t in [(axs[0], np.ones((4, 4)), 'BERT (encoder): sees both sides'), (axs[1], np.tril(np.ones((4, 4))), 'GPT (decoder): sees only the past')]:
    from matplotlib.colors import ListedColormap; ax_.imshow(M, cmap=ListedColormap(['#FBEAEA', LIGHT]), vmin=0, vmax=1); ax_.set_xticks(range(4)); ax_.set_xticklabels(toks if 'BERT' in t else ['I', 'love', 'green', 'tea'])
    ax_.set_yticks(range(4)); ax_.set_yticklabels(toks if 'BERT' in t else ['I', 'love', 'green', 'tea']); ax_.set_title(t)
    for (i, j), v in np.ndenumerate(M): ax_.text(j, i, 'v' if v else 'x', ha='center', va='center', color=NAVY if v else RED, fontweight='bold')
fig.tight_layout(); save(fig, 'c26_bert_gpt.png')
reg(26, "The most important practical idea", 'c26_bert_gpt.png', 'Which words each position may look at (v = allowed). BERT sees the whole sentence and fills in masked words; GPT sees only earlier words and predicts the next one.')

# ---------------- Chapter 27 ----------------
img = load_sample_image('china.jpg')[::2, ::2][:208, 50:362]; ph, pw = 52, 78
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[1.2, 2.2]))
axs[0].imshow(img); axs[0].set_xticks([]); axs[0].set_yticks([]); axs[0].set_title('Image cut into 16 patches')
for k in range(1, 4): axs[0].axhline(k * ph, c='white', lw=2); axs[0].axvline(k * pw, c='white', lw=2)
axs[1].axis('off'); axs[1].set_xlim(0, 17); axs[1].set_ylim(0, 4); axs[1].set_title('...flattened into a sequence of tokens')
for k in range(16):
    r, c = divmod(k, 4); axs[1].imshow(img[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw], extent=(k + 0.55, k + 1.45, 1.6, 3.0))
    axs[1].text(k + 1, 1.2, str(k + 1), ha='center', fontsize=8)
axs[1].set_aspect('auto'); axs[1].text(8.5, 0.3, 'each patch -> linear embedding + position encoding -> Transformer encoder', ha='center', fontsize=9, color=NAVY)
save(fig, 'c27_vit.png')
reg(27, "So far a Transformer has only read text", 'c27_vit.png', 'The Vision Transformer idea: cut the image into patches and feed them to a Transformer as if each patch were a word.')

fig, ax = diagram(9, 3.0, ylim=(0, 3.4))
box(ax, 1.0, 2.4, 'User\nquestion', 1.4, 0.8, fc=ORANGE, ec=ORANGE, tc='white', bold=True)
box(ax, 3.5, 2.4, 'Retriever\n(search embeddings)', 2.0, 0.8); box(ax, 3.5, 0.7, 'Document store\n(your knowledge)', 2.0, 0.8, fc='#E3F1EC', ec=TEAL)
box(ax, 6.3, 2.4, 'Prompt =\nquestion + top passages', 2.2, 0.8, fc=PALE, ec=ORANGE); box(ax, 8.6, 2.4, 'LLM\n(generator)', 1.4, 0.8, fc=NAVY, ec=NAVY, tc='white', bold=True)
box(ax, 8.6, 0.7, 'Grounded\nanswer', 1.4, 0.8, fc=GREEN, ec=GREEN, tc='white', bold=True)
arrow(ax, (1.7, 2.4), (2.5, 2.4)); arrow(ax, (3.3, 1.1), (3.3, 2.0)); arrow(ax, (3.7, 2.0), (3.7, 1.1)); arrow(ax, (4.5, 2.4), (5.2, 2.4)); arrow(ax, (7.4, 2.4), (7.9, 2.4)); arrow(ax, (8.6, 2.0), (8.6, 1.1))
save(fig, 'c27_rag.png')
reg(27, "It is worth knowing what goes into building one", 'c27_rag.png', 'Retrieval Augmented Generation: fetch the most relevant passages first, then give them to the LLM with the question so its answer rests on real sources.')

# ---------------- Chapter 28 ----------------
fig, ax = diagram(8.8, 3.2, ylim=(0, 3.6))
box(ax, 0.9, 2.6, 'Random\nnoise z', 1.3, 0.8, fc='#EEEEEE', ec=GREY); box(ax, 3.0, 2.6, 'Generator G', 1.7, 0.8, fc=ORANGE, ec=ORANGE, tc='white', bold=True)
box(ax, 5.2, 2.6, 'Fake image', 1.5, 0.7, fc=PALE, ec=ORANGE); box(ax, 5.2, 0.9, 'Real image', 1.5, 0.7, fc='#E3F1EC', ec=TEAL)
box(ax, 7.5, 1.75, 'Discriminator D', 1.9, 0.8, fc=NAVY, ec=NAVY, tc='white', bold=True)
arrow(ax, (1.55, 2.6), (2.15, 2.6)); arrow(ax, (3.85, 2.6), (4.45, 2.6)); arrow(ax, (5.95, 2.5), (6.6, 1.95)); arrow(ax, (5.95, 1.0), (6.6, 1.55))
ax.text(9.0, 1.75, 'real\nor\nfake?', va='center', fontsize=9, fontweight='bold', color=NAVY)
arrow(ax, (7.5, 2.15), (3.2, 3.0), 'feedback: how to fool D better', rad=0.25, color=RED, ls='--', toff=(0.6, 0.65))
save(fig, 'c28_gan.png')
reg(28, "That contest is exactly what a GAN", 'c28_gan.png', 'A GAN: the generator turns noise into fakes, the discriminator tries to tell fakes from real data, and each one improves by competing with the other.')

x = np.linspace(-4, 9, 500); pr = norm(4, 1).pdf(x)
fig, axs = plt.subplots(1, 3, figsize=(11, 2.9), sharey=True)
for ax_, (mu, sd, t) in zip(axs, [(-1, 1.5, 'Early training'), (2, 1.2, 'Mid training'), (4, 1.0, 'Equilibrium')]):
    pg = norm(mu, sd).pdf(x); ax_.plot(x, pr, c=TEAL, lw=2, label='real data'); ax_.plot(x, pg, c=ORANGE, lw=2, label='generator')
    ax_.plot(x, pr / (pr + pg + 1e-12) * 0.45, c=NAVY, ls='--', label='best D(x) (scaled)'); ax_.set_title(t); ax_.set_xlabel('x')
axs[0].legend(frameon=False, fontsize=8); fig.tight_layout()
save(fig, 'c28_training.png')
reg(28, "Because the two networks want opposite", 'c28_training.png', "Illustration of GAN training in one dimension. As the generator's distribution moves onto the real one, the best possible discriminator falls to 0.5 everywhere: it can no longer tell them apart.")

fig, ax = diagram(10, 2.6, ylim=(0, 3))
st = [('noise\n100', 0.5, LIGHT), ('4x4x512', 0.9, PALE), ('8x8x256', 1.25, PALE), ('16x16x128', 1.6, PALE), ('32x32x64', 1.95, PALE), ('64x64x3\nimage', 2.3, ORANGE)]
for i, (t, h, fc) in enumerate(st):
    box(ax, 0.8 + i * 1.75, 1.5, t, 1.2, h, fc=fc, ec=ORANGE if i else NAVY, fs=8.5, bold=i == 5, tc='white' if i == 5 else 'black')
    if i: arrow(ax, (0.8 + (i - 1) * 1.75 + 0.62, 1.5), (0.8 + i * 1.75 - 0.62, 1.5))
ax.text(5.2, 0.05, 'each arrow: transposed convolution doubles width and height', ha='center', fontsize=9, color=NAVY)
save(fig, 'c28_dcgan.png')
reg(28, "The generator has a shape problem", 'c28_dcgan.png', 'A DCGAN generator grows a short noise vector into a full image, using transposed convolutions that double the size at each step.')

# ---------------- Chapter 29 ----------------
fig, ax = diagram(8.8, 3.0, ylim=(0, 3.4))
box(ax, 0.9, 2.5, 'noise z', 1.2, 0.6, fc='#EEEEEE', ec=GREY); box(ax, 0.9, 1.4, "condition C\n('bird')", 1.3, 0.75, fc=GREEN, ec=GREEN, tc='white', bold=True)
box(ax, 3.2, 2.0, 'Generator', 1.5, 0.8, fc=ORANGE, ec=ORANGE, tc='white', bold=True); box(ax, 5.4, 2.0, 'fake bird', 1.4, 0.6, fc=PALE, ec=ORANGE)
box(ax, 7.7, 1.4, 'Discriminator\n(sees C too)', 1.8, 0.85, fc=NAVY, ec=NAVY, tc='white', bold=True)
arrow(ax, (1.5, 2.5), (2.45, 2.15)); arrow(ax, (1.55, 1.4), (2.45, 1.85)); arrow(ax, (3.95, 2.0), (4.7, 2.0)); arrow(ax, (6.1, 1.95), (6.8, 1.6))
arrow(ax, (1.55, 1.2), (6.8, 1.2), rad=0.15, color=GREEN, ls='--')
save(fig, 'c29_cgan.png')
reg(29, "The fix is simple in principle", 'c29_cgan.png', 'A conditional GAN: both networks also receive the condition C, so you can ask for a particular kind of output.')

fig, ax = diagram(8, 3.4, ylim=(0, 3.8))
box(ax, 1.3, 1.9, 'Horse photo\nx', 1.6, 0.9, fc='#E3F1EC', ec=TEAL, bold=True); box(ax, 6.7, 1.9, 'Zebra version\nG(x)', 1.8, 0.9, fc=PALE, ec=ORANGE, bold=True)
arrow(ax, (2.1, 2.35), (5.8, 2.35), 'G: horse -> zebra', rad=-0.3, toff=(0, 0.75)); arrow(ax, (5.8, 1.45), (2.1, 1.45), 'F: zebra -> horse', rad=-0.3, toff=(0, -0.75))
ax.text(4.0, 0.15, 'cycle loss: F(G(x)) should equal x', ha='center', color=RED, fontweight='bold')
save(fig, 'c29_cyclegan.png')
reg(29, "CycleGAN removes that requirement", 'c29_cyclegan.png', 'CycleGAN uses two generators. Translating a picture there and back should give the original picture, and that cycle loss replaces the need for paired examples.')

# ---------------- Chapter 30 ----------------
fig, axs = plt.subplots(2, 1, figsize=(8.6, 4.8))
for ax_, vae in zip(axs, (False, True)):
    ax_.axis('off'); ax_.set_xlim(0, 10); ax_.set_ylim(0, 2)
    box(ax_, 0.8, 1, 'input\nimage', 1.1, 0.9, fc=LIGHT); ax_.add_patch(plt.Polygon([[1.6, 1.6], [3.4, 1.25], [3.4, 0.75], [1.6, 0.4]], fc=PALE, ec=ORANGE)); ax_.text(2.5, 1, 'encoder', ha='center', va='center')
    if vae:
        box(ax_, 4.2, 1.45, 'mean mu', 1.1, 0.45, fs=8); box(ax_, 4.2, 0.55, 'std sigma', 1.1, 0.45, fs=8)
        box(ax_, 5.6, 1, 'z = mu +\nsigma x noise', 1.3, 0.8, fc=ORANGE, ec=ORANGE, tc='white', fs=8, bold=True); arrow(ax_, (4.75, 1.4), (4.95, 1.15)); arrow(ax_, (4.75, 0.6), (4.95, 0.85))
    else:
        box(ax_, 4.9, 1, 'code z\n(bottleneck)', 1.3, 0.7, fc=ORANGE, ec=ORANGE, tc='white', fs=8, bold=True)
    ax_.add_patch(plt.Polygon([[6.5, 1.25], [8.3, 1.6], [8.3, 0.4], [6.5, 0.75]], fc='#E3F1EC', ec=TEAL)); ax_.text(7.4, 1, 'decoder', ha='center', va='center')
    box(ax_, 9.25, 1, 'rebuilt\nimage', 1.1, 0.9, fc=LIGHT); ax_.set_title('Variational autoencoder: the code is a distribution you can sample' if vae else 'Autoencoder: squeeze, then rebuild', color=NAVY, fontweight='bold')
save(fig, 'c30_ae_vae.png')
reg(30, "The Variational Autoencoder makes one change", 'c30_ae_vae.png', 'Top: an autoencoder squeezes an image into a code and rebuilds it. Bottom: a VAE encodes a mean and spread instead, and samples the code, so new codes give new images.')

g = load_sample_image('flower.jpg').mean(2)[::6, ::6] / 255.0; x0 = (g - 0.5) * 2
betas = np.linspace(1e-4, 0.02, 1000); abar = np.cumprod(1 - betas); rng = np.random.default_rng(0); eps = rng.normal(0, 1, x0.shape)
fig, axs = plt.subplots(1, 5, figsize=(11, 2.6))
for ax_, t in zip(axs, [0, 50, 200, 500, 999]):
    xt = x0 if t == 0 else np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps
    ax_.imshow(xt, cmap='gray', vmin=-2, vmax=2); ax_.set_xticks([]); ax_.set_yticks([]); ax_.set_title(f't = {t + 1 if t else 0}' + (f'\nsignal kept {np.sqrt(abar[t]):.2f}' if t else '\noriginal'), fontsize=9)
fig.tight_layout(); save(fig, 'c30_diffusion.png')
reg(30, "Diffusion models, the engine", 'c30_diffusion.png', 'The forward diffusion process on a real photo (1,000 steps, linear noise schedule). The model learns to run this in reverse, turning pure noise back into an image.')

# ---------------- Chapter 31 ----------------
stages = ['Problem\ndefinition', 'Data\nengineering', 'Model\ndevelopment', 'Validation', 'Versioning', 'Deployment', 'Monitoring\n& logging', 'CI/CD\npipelines', 'Retraining']
fig, ax = diagram(7, 6, xlim=(-4, 4), ylim=(-3.6, 3.6)); ax.set_aspect('equal')
ang = np.pi / 2 - np.arange(9) * 2 * np.pi / 9; R = 2.6
for k, (t, a) in enumerate(zip(stages, ang)):
    box(ax, R * np.cos(a), R * np.sin(a), t, 1.55, 0.75, fc=ORANGE if k == 8 else LIGHT, ec=ORANGE if k == 8 else NAVY, fs=8, tc='white' if k == 8 else 'black', bold=k == 8)
    a2 = ang[(k + 1) % 9]; m1, m2 = a - 0.27, a2 + 0.27
    arrow(ax, (R * np.cos(m1), R * np.sin(m1)), (R * np.cos(m2), R * np.sin(m2)), rad=-0.15, color=GREY)
ax.text(0, 0, 'The ML\nlifecycle\nis a loop', ha='center', va='center', fontsize=12, fontweight='bold', color=NAVY)
save(fig, 'c31_lifecycle.png')
reg(31, "The lifecycle has nine stages", 'c31_lifecycle.png', 'The nine stages of the ML lifecycle. Retraining feeds back into the start, which turns deployment into a loop rather than a one-off task.')

rng = np.random.default_rng(1); train = rng.normal(50, 10, 5000); prod = rng.normal(58, 12, 5000)
edges = np.percentile(train, np.linspace(0, 100, 11)); edges[0], edges[-1] = -1e9, 1e9
e_, a_ = np.histogram(train, edges)[0] / 5000, np.histogram(prod, edges)[0] / 5000; psi = np.sum((a_ - e_) * np.log(a_ / e_))
fig, ax = plt.subplots(figsize=(6.4, 3.3)); bins = np.linspace(10, 110, 50)
ax.hist(train, bins, color=NAVY, alpha=0.5, label='Training data'); ax.hist(prod, bins, color=ORANGE, alpha=0.5, label='Live data this month')
ax.set_xlabel('Feature value (e.g. average basket size)'); ax.set_ylabel('Count'); ax.legend(frameon=False); ax.set_title(f'Data drift: PSI = {psi:.2f} (above 0.25 means significant drift)')
save(fig, 'c31_drift.png')
reg(31, "Why go to all this trouble?", 'c31_drift.png', 'Drift: the live data has moved away from the data the model was trained on. The Population Stability Index (PSI) puts a number on how far.')

# ---------------- Chapter 32 ----------------
fig, ax = diagram(9, 2.6, ylim=(0, 3))
opts = [('Virtual machine', 'you manage the OS\nany workload'), ('GPU / TPU node', 'heavy training\njobs'), ('Containers +\nKubernetes', 'portable apps\nautoscaling'), ('Serverless\nfunction', 'small, quick calls\npay per call')]
for i, (t, sub) in enumerate(opts):
    box(ax, 1.2 + i * 2.4, 1.9, t, 2.0, 0.8, fc=LIGHT, bold=True, fs=8.5); ax.text(1.2 + i * 2.4, 0.95, sub, ha='center', va='center', fontsize=8.5, color=GREY)
ax.annotate('', xy=(9.4, 0.2), xytext=(0.2, 0.2), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=2)); ax.text(4.8, 0.32, 'less to manage, less control', ha='center', color=ORANGE, fontsize=9)
save(fig, 'c32_compute.png')
reg(32, "Pick the compute to match the job", 'c32_compute.png', 'Cloud compute options, from full control (a virtual machine) to almost nothing to manage (serverless).')

fig, ax = diagram(9.6, 3.6, ylim=(0, 4.2))
box(ax, 0.8, 2.1, 'Users /\napps', 1.2, 0.8, fc=ORANGE, ec=ORANGE, tc='white', bold=True)
ax.add_patch(FancyBboxPatch((2.0, 0.3), 7.4, 3.6, boxstyle='round,pad=0.05', fc='#F7F9FC', ec=TEAL, lw=1.5, ls='--')); ax.text(2.2, 3.65, 'Virtual Private Cloud', color=TEAL, fontweight='bold', fontsize=9)
box(ax, 3.2, 2.1, 'Load\nbalancer', 1.3, 0.8)
ax.add_patch(FancyBboxPatch((4.4, 0.8), 2.3, 2.6, boxstyle='round,pad=0.05', fc='white', ec=NAVY)); ax.text(5.55, 3.15, 'autoscaling group', ha='center', fontsize=8, color=NAVY)
for k, yy in enumerate([2.6, 1.9, 1.2]): box(ax, 5.55, yy, f'model container {k + 1}', 1.9, 0.5, fc=PALE, ec=ORANGE, fs=8)
box(ax, 8.2, 2.7, 'Model registry /\nobject storage', 1.7, 0.75, fc='#E3F1EC', ec=TEAL, fs=8); box(ax, 8.2, 1.3, 'Monitoring\n(logs, metrics)', 1.7, 0.75, fc='#E3F1EC', ec=TEAL, fs=8)
arrow(ax, (1.4, 2.1), (2.55, 2.1)); arrow(ax, (3.85, 2.1), (4.55, 2.1)); arrow(ax, (7.35, 2.7), (6.75, 2.4), ls='--', color=GREY); arrow(ax, (6.75, 1.4), (7.35, 1.3), ls='--', color=GREY)
save(fig, 'c32_architecture.png')
reg(32, "Then there is keeping it safe", 'c32_architecture.png', 'A typical model-serving setup: requests pass through a load balancer to an autoscaling group of model containers, all inside a private network.')

h = np.arange(0, 24, 0.25); load = 200 + 900 * np.exp(-((h - 13) / 3.2) ** 2) + 300 * np.exp(-((h - 20) / 1.5) ** 2)
inst = np.maximum(2, np.ceil(load / (150 * 0.7)))
fig, ax = plt.subplots(figsize=(6.6, 3.4)); ax.plot(h, load, c=NAVY, lw=2, label='Requests per second'); ax.set_xlabel('Hour of day'); ax.set_ylabel('Requests / second', color=NAVY)
ax2 = ax.twinx(); ax2.step(h, inst, c=ORANGE, lw=2, where='post', label='Instances running'); ax2.set_ylabel('Instances', color=ORANGE); ax2.spines['right'].set_visible(True)
ax.set_title('Autoscaling: keep each instance at about 70% load')
save(fig, 'c32_autoscaling.png')
reg(32, "Last comes scaling and cost", 'c32_autoscaling.png', 'Autoscaling over one day. Each instance handles 150 requests per second; the scaler adds or removes instances to keep them near 70% busy, with a minimum of two.')

# ---------------- Chapter 33 ----------------
fig, ax = diagram(7, 3.0, ylim=(0, 3.4))
box(ax, 2, 1.7, 'Agent\n(policy pi)', 1.9, 1.0, fc=ORANGE, ec=ORANGE, tc='white', bold=True, fs=10)
box(ax, 8, 1.7, 'Environment\n(transition model)', 2.4, 1.0, fc=TEAL, ec=TEAL, tc='white', bold=True, fs=10)
arrow(ax, (2.6, 2.25), (7.4, 2.25), 'action a', rad=-0.35, toff=(0, 0.75), fs=10)
arrow(ax, (7.4, 1.15), (2.6, 1.15), "next state s' and reward R", rad=-0.35, toff=(0, -0.75), fs=10)
save(fig, 'c33_agent_env.png')
reg(33, "The vocabulary is small", 'c33_agent_env.png', 'The agent picks an action from its policy; the environment returns the next state and a reward. The goal is the largest total reward over time.')

W_, T1, T2 = (2, 2), (4, 3), (4, 2); states = [(x, y) for x in range(1, 5) for y in range(1, 4) if (x, y) != W_]
A = {'^': (0, 1), 'v': (0, -1), '<': (-1, 0), '>': (1, 0)}; side = {'^': '<>', 'v': '<>', '<': '^v', '>': '^v'}
def mv(s, a):
    n = (s[0] + A[a][0], s[1] + A[a][1]); return s if n == W_ or not (1 <= n[0] <= 4 and 1 <= n[1] <= 3) else n
def outc(s, a): return [(0.8, mv(s, a))] + [(0.1, mv(s, b)) for b in side[a]]
U = {s: 0.0 for s in states}; U[T1], U[T2] = 1.0, -1.0; hist = {s: [] for s in [(1, 1), (3, 3), (1, 3), (4, 1)]}
for it in range(30):
    N = dict(U)
    for s in states:
        if s in (T1, T2): continue
        N[s] = -0.04 + max(sum(p * U[n] for p, n in outc(s, a)) for a in A)
    U = N
    for s in hist: hist[s].append(U[s])
pol = {s: max(A, key=lambda a: sum(p * U[n] for p, n in outc(s, a))) for s in states if s not in (T1, T2)}
fig, ax = plt.subplots(figsize=(6, 4.2)); arrows_ = {'^': '↑', 'v': '↓', '<': '←', '>': '→'}
for (x, y) in [(x, y) for x in range(1, 5) for y in range(1, 4)]:
    fc = '#555555' if (x, y) == W_ else '#D4EDDA' if (x, y) == T1 else '#F8D7DA' if (x, y) == T2 else 'white'
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc=fc, ec=NAVY, lw=1.5))
    if (x, y) in pol: ax.text(x, y + 0.15, f'{U[(x, y)]:.3f}', ha='center', fontsize=10); ax.text(x, y - 0.22, arrows_[pol[(x, y)]], ha='center', fontsize=18, color=ORANGE)
    elif (x, y) != W_: ax.text(x, y, f'{U[(x, y)]:+.0f}', ha='center', va='center', fontsize=14, fontweight='bold')
ax.set_xlim(0.5, 4.5); ax.set_ylim(0.5, 3.5); ax.set_aspect('equal'); ax.set_xticks([1, 2, 3, 4]); ax.set_yticks([1, 2, 3]); ax.set_title('Utilities and the optimal policy (R = -0.04, gamma = 1)')
save(fig, 'c33_grid.png')
reg(33, "The Bellman equation ties it all", 'c33_grid.png', 'The 4x3 grid world solved by value iteration: the utility of every square and the best action from it. Note how the agent near the -1 square heads away from it rather than risk slipping in.')

fig, ax = plt.subplots(figsize=(6.2, 3.3))
for s, c in zip(hist, [NAVY, ORANGE, TEAL, RED]): ax.plot(range(1, 31), hist[s], c=c, lw=2, label=f'U{s}')
ax.set_xlabel('Iteration'); ax.set_ylabel('Utility estimate'); ax.legend(frameon=False, fontsize=8.5); ax.set_title('Value iteration converges in about 20 iterations')
save(fig, 'c33_convergence.png')
reg(33, "The Bellman equation ties it all", 'c33_convergence.png', 'How the utility estimates of four squares settle as value iteration repeats the Bellman update.')

# ---------------- Chapter 34 ----------------
fig, ax = diagram(7, 2.8, ylim=(0, 3.2))
box(ax, 2.2, 1.6, 'Policy evaluation\nsolve for U under pi', 2.6, 1.0, fc=LIGHT, bold=True, fs=9); box(ax, 7.5, 1.6, 'Policy improvement\npick best action\nper state', 2.6, 1.1, fc=PALE, ec=ORANGE, bold=True, fs=9)
arrow(ax, (3.0, 2.15), (6.6, 2.15), 'U', rad=-0.3, toff=(0, 0.6)); arrow(ax, (6.6, 1.05), (3.0, 1.05), 'new pi', rad=-0.3, toff=(0, -0.6))
ax.text(5, 0.05, 'stop when the policy no longer changes', ha='center', fontsize=9, color=RED)
save(fig, 'c34_policy_iter.png')
reg(34, "First, a second way to solve an MDP", 'c34_policy_iter.png', 'Policy iteration alternates two steps until the policy stops changing.')

fig, ax = plt.subplots(figsize=(8.4, 3.2))
for c in range(12):
    for r in range(4):
        fc = '#F8D7DA' if (r == 3 and 1 <= c <= 10) else '#D4EDDA' if (r == 3 and c == 11) else '#E3F1EC' if (r == 3 and c == 0) else 'white'
        ax.add_patch(Rectangle((c, 3 - r), 1, 1, fc=fc, ec=GREY, lw=0.8))
ax.text(0.5, 0.5, 'S', ha='center', va='center', fontweight='bold'); ax.text(11.5, 0.5, 'G', ha='center', va='center', fontweight='bold'); ax.text(6, 0.5, 'THE CLIFF  (-100)', ha='center', va='center', color=RED, fontweight='bold')
q = [(0.5, 0.5), (0.5, 1.5), (11.5, 1.5), (11.5, 0.5)]; sa = [(0.5, 0.5), (0.5, 3.5), (11.5, 3.5), (11.5, 0.5)]
ax.plot(*zip(*q), c=ORANGE, lw=3, label='Q-learning: shortest, along the edge (13 steps)'); ax.plot(*zip(*sa), c=NAVY, lw=3, ls='--', label='SARSA: safe, along the top (17 steps)')
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.set_aspect('equal'); ax.axis('off'); ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8.5)
save(fig, 'c34_cliff.png')
reg(34, "Two active TD methods are worth knowing", 'c34_cliff.png', 'The greedy paths learned in Example 2. Q-learning hugs the cliff edge; SARSA, which accounts for its own random exploration, keeps well away.')

bins = ['1-50', '51-100', '101-200', '201-350', '351-500']; sar = [-109.2, -34.3, -29.1, -27.2, -26.9]; ql = [-120.2, -52.6, -46.1, -50.5, -51.7]
fig, ax = plt.subplots(figsize=(6.4, 3.3)); xi = np.arange(5)
ax.bar(xi - 0.2, sar, 0.4, color=NAVY, label='SARSA'); ax.bar(xi + 0.2, ql, 0.4, color=ORANGE, label='Q-learning')
ax.set_xticks(xi); ax.set_xticklabels(bins); ax.set_xlabel('Episodes'); ax.set_ylabel('Average reward per episode'); ax.legend(frameon=False, loc='lower right'); ax.set_title('While still exploring, SARSA earns more')
save(fig, 'c34_rewards.png')
reg(34, "Two active TD methods are worth knowing", 'c34_rewards.png', 'Average reward per episode from Example 2 (20 runs). Q-learning keeps falling off the cliff during exploration, so SARSA collects more reward.')

fig, ax = diagram(8.6, 3.2, ylim=(0, 3.6))
box(ax, 2.0, 2.6, 'Actor\n(policy network)', 2.2, 0.9, fc=ORANGE, ec=ORANGE, tc='white', bold=True); box(ax, 2.0, 0.8, 'Critic\n(Q-network)', 2.2, 0.9, fc=NAVY, ec=NAVY, tc='white', bold=True)
box(ax, 7.0, 1.7, 'Environment', 2.0, 0.9, fc=TEAL, ec=TEAL, tc='white', bold=True)
arrow(ax, (3.1, 2.6), (6.0, 1.95), 'action'); arrow(ax, (6.0, 1.45), (3.1, 0.85), 'state, reward')
arrow(ax, (2.0, 1.25), (2.0, 2.15), 'TD error: "better or worse\nthan expected?"', toff=(-1.35, 0), color=RED)
save(fig, 'c34_actor_critic.png')
reg(34, "Policy gradient methods skip values", 'c34_actor_critic.png', 'Actor-critic: the actor chooses actions, the critic scores them with the TD error, and that score tells the actor which way to improve.')

fig, ax = diagram(10, 3.3, ylim=(0, 3.9))
for i, (t, d) in enumerate([('1. Selection', 'follow promising\nbranches'), ('2. Expansion', 'add a new\nnode'), ('3. Simulation', 'play a random\ngame to the end'), ('4. Backup', 'update win counts\nback up the path')]):
    box(ax, 1.3 + i * 2.5, 2.6, t, 2.0, 0.6, fc=NAVY, ec=NAVY, tc='white', bold=True, fs=9); ax.text(1.3 + i * 2.5, 1.8, d, ha='center', va='center', fontsize=8.5)
    if i < 3: arrow(ax, (2.35 + i * 2.5, 2.6), (2.75 + i * 2.5, 2.6))
    cx = 1.3 + i * 2.5; ax.add_patch(Circle((cx, 1.15), 0.12, color=GREY)); 
    for dx in (-0.4, 0.4): ax.plot([cx, cx + dx], [1.15, 0.65], c=GREY); ax.add_patch(Circle((cx + dx, 0.65), 0.1, color=ORANGE if (i == 0 and dx > 0) else GREY))
    if i >= 1: ax.plot([cx + 0.4, cx + 0.6], [0.65, 0.25], c=GREEN, lw=2); ax.add_patch(Circle((cx + 0.6, 0.25), 0.09, color=GREEN))
    if i == 2: ax.plot([cx + 0.6, cx + 0.9], [0.25, 0.02], c=GREEN, ls=':', lw=2)
arrow(ax, (8.8, 2.95), (1.3, 2.95), 'repeat thousands of times', rad=0.15, color=GREY, toff=(0, 0.3))
save(fig, 'c34_mcts.png')
reg(34, "For big decision trees such as board games", 'c34_mcts.png', 'The four phases of Monte Carlo Tree Search, repeated many times; the move with the best record at the root is then played.')
