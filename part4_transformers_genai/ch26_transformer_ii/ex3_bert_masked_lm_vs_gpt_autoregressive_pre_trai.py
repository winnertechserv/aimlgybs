"""
Chapter 26, Example 3: BERT Masked LM vs GPT Autoregressive Pre-Training
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This contrasts the two pre-training objectives directly. BERT masks about 15% of tokens
and predicts them using both left and right context; GPT predicts each next word from
the preceding words only. The final section computes the cross-entropy loss for a single
prediction, showing how a confident wrong answer is penalised far more heavily than an
uncertain one.

Run it:           python ex3_bert_masked_lm_vs_gpt_autoregressive_pre_trai.py
Expected output:  expected_output/ex3_bert_masked_lm_vs_gpt_autoregressive_pre_trai.txt
"""
# Chapter 26 - Example 3: BERT Masked LM vs GPT Autoregressive pre-training
import numpy as np
np.random.seed(3)

sentence = "the quick brown fox jumps over the lazy dog".split()
T = len(sentence)

print("=== Pre-Training Objectives: BERT vs GPT ===\n")
print("Both pre-train on huge UNLABELED text, then fine-tune on a small")
print("labeled dataset for a downstream task.\n")
print(f"Sentence ({T} tokens): {' '.join(sentence)}\n")

# ---------- BERT: Masked Language Modelling ----------
print("--- BERT (encoder pre-training): Masked Language Modelling ---")
print("Randomly mask ~15% of tokens; the model predicts them using BOTH")
print("left and right context (bidirectional).\n")
n_mask = max(1, int(0.15 * T))
mask_idx = sorted(int(i) for i in np.random.choice(T, n_mask, replace=False))
masked_sentence = [("[MASK]" if i in mask_idx else w) for i, w in enumerate(sentence)]
print(f"  Masked positions : {mask_idx}")
print(f"  Model input      : {' '.join(masked_sentence)}")
print(f"  Prediction target: {[sentence[i] for i in mask_idx]}")
print("  Context used     : ALL other tokens (both directions)\n")

print("  Second BERT objective - Next Sentence Prediction (NSP):")
print("    Input : [CLS] the quick brown fox ... [SEP] it was very fast [SEP]")
print("    Target: IsNext = True/False\n")

# ---------- GPT: Autoregressive LM ----------
print("--- GPT (decoder pre-training): Autoregressive Language Modelling ---")
print("Predict the NEXT word using only the PRECEDING words (unidirectional).\n")
print(f"  {'t':>2}  {'Input (context)':<45} {'Predict'}")
print("  " + "-" * 66)
for t in range(1, min(T, 6)):
    ctx = " ".join(sentence[:t])
    print(f"  {t:>2}  {ctx:<45} -> {sentence[t]}")
print("  ...")

# ---------- Cross-entropy loss for one prediction ----------
print("\n=== Cross-Entropy Loss for One Prediction ===\n")
vocab = ["fox", "dog", "cat", "car", "tree"]
logits = np.array([3.2, 1.1, 0.8, -0.5, 0.2])
e = np.exp(logits - logits.max()); probs = e / e.sum()
target = 0   # correct word is "fox"

print(f"Vocabulary      : {vocab}")
print(f"Model logits    : {logits}")
print(f"Softmax probs   : {np.round(probs, 4)}")
print(f"True next word  : '{vocab[target]}' (one-hot target)")
loss = -np.log(probs[target])
print(f"Cross-entropy loss = -log(p_correct) = -log({probs[target]:.4f}) = {loss:.4f}")

print("\nIf the model were confident and WRONG (p_correct = 0.01):")
print(f"  loss = -log(0.01) = {-np.log(0.01):.4f}   (heavily penalised)")
print("If the model were perfect (p_correct = 1.0):")
print(f"  loss = -log(1.00) = {-np.log(1.0):.4f}")

print("\n=== Summary ===")
print("  BERT : encoder-only,  bidirectional, Masked LM + NSP")
print("  GPT  : decoder-only,  autoregressive (masked self-attention)")
print("  T5   : encoder-decoder, text-to-text transfer")
print("  ViLBERT: encoder pre-training for visual-linguistic (image+text) data")
