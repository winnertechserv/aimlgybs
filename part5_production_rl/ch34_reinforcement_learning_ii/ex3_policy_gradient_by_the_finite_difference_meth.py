"""
Chapter 34, Example 3: Policy Gradient by the Finite-Difference Method
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This follows the finite-difference recipe exactly. A simple policy starts out preferring
the wrong direction in a corridor, so its total reward J is negative. Each iteration it
tries a dozen small random changes to its weights, measures how the reward moved, fits a
gradient with the regression formula (D^T D)^-1 D^T y, and steps uphill. Without ever
learning a value function, J climbs from -0.75 to +0.84. The closing note explains how
actor-critic methods speed this up.

Run it:           python ex3_policy_gradient_by_the_finite_difference_meth.py
Expected output:  expected_output/ex3_policy_gradient_by_the_finite_difference_meth.txt
"""
# Chapter 34 - Example 3: Policy Gradient by the Finite-Difference Method
import numpy as np
rng = np.random.default_rng(0)

print("=== Policy Gradient: the Finite-Difference Method ===\n")
print("Value-based methods learn U(s) or Q(s,a) and then pick actions from it.")
print("Policy gradient methods skip that: they adjust the policy's parameters")
print("theta DIRECTLY, in the direction that raises the total reward J.\n")
print("The finite-difference recipe:")
print("  1. Nudge the weights m times by small random amounts  delta_theta_i")
print("  2. Run a roll-out of H moves with each nudged policy, measure delta_J_i")
print("  3. Stack nudges as rows of D and reward changes into y, then solve")
print("        grad J = (D^T D)^-1 D^T y        (the linear-regression formula)")
print("  4. Gradient ASCENT:  theta <- theta + eta * grad J\n")

# ---- A corridor: states 0..6, start at 3. Reaching 6 pays +1, reaching 0 costs -1.
N, START, H, GAMMA = 7, 3, 20, 0.95

def features(s):
    return np.array([1.0, (s - 3) / 3.0])           # bias + position

def p_right(theta, s):
    return 1.0 / (1.0 + np.exp(-features(s) @ theta))

def rollout_return(theta):
    s, J = START, 0.0
    for t in range(H):
        s += 1 if rng.random() < p_right(theta, s) else -1
        r = 1.0 if s == N - 1 else (-1.0 if s == 0 else -0.01)
        J += (GAMMA ** t) * r
        if s in (0, N - 1):
            break
    return J

def J_estimate(theta, n=300):
    return np.mean([rollout_return(theta) for _ in range(n)])

print("Environment: a corridor of 7 cells, start in the middle (cell 3).")
print("Right end = +1, left end = -1, every other step = -0.01, horizon H=20.")
print("Policy: P(move right | s) = sigmoid(theta_0 + theta_1 * position)\n")

theta = np.array([-1.0, 0.0])                   # starts out preferring LEFT (wrong way)
m, sigma, eta = 12, 0.3, 0.8

print(f"{'iter':>5}{'theta_0':>10}{'theta_1':>10}{'P(right) at start':>20}{'J':>10}")
print("-" * 57)
for it in range(13):
    J0 = J_estimate(theta)
    if it % 2 == 0 or it == 12:
        print(f"{it:>5}{theta[0]:>10.3f}{theta[1]:>10.3f}{p_right(theta, START):>20.3f}{J0:>10.3f}")
    D = sigma * rng.standard_normal((m, 2))            # m perturbations (rows of D)
    y = np.array([J_estimate(theta + d) - J0 for d in D])
    grad = np.linalg.solve(D.T @ D, D.T @ y)           # (D^T D)^-1 D^T y
    theta = theta + eta * grad

print("\nWhat happened:")
print("  The policy began by preferring the WRONG direction (P(right) = 0.27),")
print("  so it usually fell off the left end and J was negative.")
print("  Nobody told it which way to go. Each iteration it just tried small")
print("  random changes, saw which ones earned more reward, fitted a gradient,")
print("  and stepped uphill. P(right) climbed to nearly 0.9 and J turned positive.\n")

print("=== From here to Actor-Critic ===")
print("  Finite differences need many roll-outs per step, which is slow.")
print("  Actor-critic methods use TWO networks instead:")
print("    the ACTOR  (policy network) chooses actions,")
print("    the CRITIC (Q-network) scores them using the TD error")
print("        delta_t = R_t + gamma*Q(s_t+1, a_t+1) - Q(s_t, a_t).")
print("  The critic's score tells the actor which way to move, one step at a time,")
print("  without waiting for whole roll-outs. AlphaGo and walking robots use this idea.")
