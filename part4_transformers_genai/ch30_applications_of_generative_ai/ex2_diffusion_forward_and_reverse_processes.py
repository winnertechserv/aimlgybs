"""
Chapter 30, Example 2: Diffusion Forward and Reverse Processes
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This builds the forward diffusion equation step by step, tracking how correlation with
the original data decays until x_T is essentially Gaussian noise. It also shows the
closed-form shortcut using alpha_bar, then runs the reverse process with a deliberately
imperfect denoiser, so you can watch the error fall from 0.78 to 0.03 as structure is
progressively restored.

Run it:           python ex2_diffusion_forward_and_reverse_processes.py
Expected output:  expected_output/ex2_diffusion_forward_and_reverse_processes.txt
"""
# Chapter 30 - Example 2: Diffusion model forward & reverse processes
import numpy as np
np.random.seed(1)
np.set_printoptions(suppress=True)

print("=== Diffusion Models: Forward and Reverse Processes ===\n")
print("Forward diffusion : slowly DESTROY structure by adding noise")
print("Reverse diffusion : learn to RESTORE structure, turning noise into data\n")
print("Forward step:")
print("  x_(t+1) = sqrt(1 - beta_t) * x_t + sqrt(beta_t) * eps ,  eps ~ N(0, I)")
print("  As T -> infinity, x_T ~ N(0, I)  (pure noise)\n")

# A tiny 1-D 'image' with clear structure
x0 = np.array([1.0, 0.8, -0.6, -1.0, 0.4])
T = 12
betas = np.linspace(0.05, 0.35, T)      # variance schedule

print(f"Original data x_0 : {np.round(x0, 3)}")
print(f"Noise schedule beta_t from {betas[0]:.2f} to {betas[-1]:.2f} over T={T} steps\n")

# ---- Forward process ----
print("--- Forward Diffusion Process (adding noise step by step) ---\n")
print(f"{'t':>3}  {'beta_t':>7}  {'x_t':<42} {'corr with x_0':>14}")
print("-" * 74)
x = x0.copy()
traj = [x.copy()]
for t in range(T):
    eps = np.random.randn(*x.shape)
    x = np.sqrt(1 - betas[t]) * x + np.sqrt(betas[t]) * eps
    traj.append(x.copy())
    if t % 3 == 0 or t == T-1:
        corr = np.corrcoef(x0, x)[0, 1]
        print(f"{t+1:>3}  {betas[t]:>7.3f}  {str(np.round(x,2)):<42} {corr:>14.4f}")

print(f"\nFinal x_T mean = {traj[-1].mean():+.3f}, std = {traj[-1].std():.3f}")
print("The structure of x_0 is essentially destroyed -> x_T is nearly N(0,I).\n")

# ---- Closed form: jump straight to any t ----
print("--- Closed-form shortcut ---\n")
alphas = 1 - betas
alpha_bar = np.cumprod(alphas)
print("Instead of looping, we can jump directly to step t:")
print("  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps\n")
print(f"{'t':>3}  {'alpha_bar_t':>12}  {'signal kept':>12}  {'noise added':>12}")
print("-" * 46)
for t in [0, 3, 6, 9, T-1]:
    print(f"{t+1:>3}  {alpha_bar[t]:>12.4f}  {np.sqrt(alpha_bar[t]):>12.4f}  {np.sqrt(1-alpha_bar[t]):>12.4f}")
print("\nSignal decays and noise grows monotonically - that is the whole schedule.")

# ---- Reverse process ----
print("\n--- Reverse Diffusion Process (denoising) ---\n")
print("A neural network (a U-Net with conv blocks, self-attention, GroupNorm,")
print("GELU/SiLU activations) predicts the noise eps at each step, and we")
print("subtract it to step back from x_t to x_(t-1).\n")

def predict_noise(x_t, t, err=0.15):
    """Stand-in for the TRAINED denoising network (imperfect, like the real one)."""
    true_eps = (x_t - np.sqrt(alpha_bar[t]) * x0) / np.sqrt(1 - alpha_bar[t])
    return true_eps + err * np.random.randn(*x_t.shape)   # network is not exact

x = traj[-1].copy()
print(f"Start from pure noise x_T : {np.round(x, 3)}")
print("Stepping backwards, each step removes a little predicted noise:\n")
print(f"{'step t':>7}  {'x_t':<40} {'err vs x_0':>11}")
print("-" * 62)
for t in reversed(range(T)):
    eps_hat = predict_noise(x, t)
    # one reverse step: remove predicted noise, keep the schedule scaling
    x0_hat = (x - np.sqrt(1 - alpha_bar[t]) * eps_hat) / np.sqrt(alpha_bar[t])
    if t > 0:
        x = np.sqrt(alpha_bar[t-1]) * x0_hat + np.sqrt(1 - alpha_bar[t-1]) * np.random.randn(*x.shape) * 0.3
    else:
        x = x0_hat
    if t in (11, 8, 5, 2, 0):
        print(f"{t:>7}  {str(np.round(x,3)):<40} {np.abs(x-x0).mean():>11.4f}")

print(f"\nRecovered x_0             : {np.round(x, 3)}")
print(f"Original  x_0             : {np.round(x0, 3)}")
print(f"Mean absolute error       : {np.abs(x - x0).mean():.4f}")
print("The reverse process progressively recovers the structure. It is not")
print("exact because the denoising network is only an approximation - which")
print("is also why real diffusion models use hundreds of small steps.")

print("\n=== DALL-E 2 text-to-image pipeline ===")
print("  1. CLIP text encoder  : caption -> text representation")
print("  2. Diffusion prior    : text encoding -> CLIP image encoding")
print("  3. Reverse diffusion  : image encoding -> final generated image")
