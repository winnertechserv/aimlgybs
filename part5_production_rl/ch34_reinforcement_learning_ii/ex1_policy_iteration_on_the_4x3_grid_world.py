"""
Chapter 34, Example 1: Policy Iteration on the 4x3 Grid World
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This solves the same grid world as Chapter 33, but with policy iteration. It starts from
a bad policy, 'always go up', and repeats two steps: work out the exact value of every
state under the current policy by solving a set of linear equations, then switch each
state to its best action. After just five rounds the policy stops changing. Look at the
last table: the utilities match the value-iteration answers from Chapter 33, so two
different methods agree on the same optimal policy.

Run it:           python ex1_policy_iteration_on_the_4x3_grid_world.py
Expected output:  expected_output/ex1_policy_iteration_on_the_4x3_grid_world.txt
"""
# Chapter 34 - Example 1: Policy Iteration on the 4x3 Grid World
import numpy as np

print("=== Policy Iteration on the 4x3 Grid World ===\n")
print("Same world as Chapter 33: wall at (2,2), +1 at (4,3), -1 at (4,2),")
print("step reward -0.04, action succeeds with p=0.8, slips sideways 0.1 each.\n")
print("Policy iteration alternates two steps until the policy stops changing:")
print("  1. Policy EVALUATION : work out U(s) for the CURRENT policy (no max)")
print("  2. Policy IMPROVEMENT: in each state, switch to the best action\n")

rows, cols = 3, 4
WALL = (2, 2)
TERMINALS = {(4, 3): 1.0, (4, 2): -1.0}
R_STEP, GAMMA = -0.04, 1.0
ACTIONS = {"Up": (0, 1), "Down": (0, -1), "Left": (-1, 0), "Right": (1, 0)}
SIDE = {"Up": ["Left", "Right"], "Down": ["Left", "Right"],
        "Left": ["Up", "Down"], "Right": ["Up", "Down"]}
ARROW = {"Up": "^", "Down": "v", "Left": "<", "Right": ">"}

states = [(c, r) for c in range(1, cols + 1) for r in range(1, rows + 1) if (c, r) != WALL]
nonterm = [s for s in states if s not in TERMINALS]
idx = {s: i for i, s in enumerate(nonterm)}

def move(s, a):
    dc, dr = ACTIONS[a]
    ns = (s[0] + dc, s[1] + dr)
    if ns == WALL or not (1 <= ns[0] <= cols and 1 <= ns[1] <= rows):
        return s
    return ns

def outcomes(s, a):
    return [(0.8, move(s, a))] + [(0.1, move(s, b)) for b in SIDE[a]]

def evaluate(policy):
    """Policy evaluation: solve the LINEAR system U = R + gamma * P_pi U exactly."""
    n = len(nonterm)
    A = np.eye(n)
    b = np.zeros(n)
    for s in nonterm:
        i = idx[s]
        for p, ns in outcomes(s, policy[s]):
            b[i] += p * R_STEP
            if ns in TERMINALS:
                b[i] += p * GAMMA * TERMINALS[ns]
            else:
                A[i, idx[ns]] -= p * GAMMA
    U_vec = np.linalg.solve(A, b)
    U = {s: U_vec[idx[s]] for s in nonterm}
    U.update(TERMINALS)
    return U

def q_value(U, s, a):
    return sum(p * (R_STEP + GAMMA * U[ns]) for p, ns in outcomes(s, a))

def show(policy):
    for r in range(rows, 0, -1):
        line = f"    row {r} |"
        for c in range(1, cols + 1):
            s = (c, r)
            if s == WALL: line += "  # "
            elif s in TERMINALS: line += f" {'+1' if TERMINALS[s] > 0 else '-1'} "
            else: line += f"  {ARROW[policy[s]]} "
        print(line)

# Start from a deliberately poor policy: always go Up
policy = {s: "Up" for s in nonterm}
print("Initial policy (always Up):")
show(policy)

for it in range(1, 20):
    U = evaluate(policy)                                   # step 1
    changed = 0
    for s in nonterm:                                      # step 2
        best = max(ACTIONS, key=lambda a: q_value(U, s, a))
        if q_value(U, s, best) > q_value(U, s, policy[s]) + 1e-12:
            policy[s] = best
            changed += 1
    print(f"\nIteration {it}: U(1,1)={U[(1,1)]:+.4f}  U(3,3)={U[(3,3)]:+.4f}  "
          f"actions changed = {changed}")
    if changed == 0:
        print("Policy is stable -> it is optimal. Stop.")
        break

U = evaluate(policy)
print("\nFinal optimal policy:")
show(policy)

print("\nFinal utilities compared with VALUE iteration from Chapter 33:")
vi = {(1, 1): 0.7053, (1, 3): 0.8116, (3, 3): 0.9178, (4, 1): 0.3878}
print(f"    {'state':<8}{'policy iter.':>14}{'value iter.':>13}")
for s, v in vi.items():
    print(f"    {str(s):<8}{U[s]:>14.4f}{v:>13.4f}")

print("\nBoth methods reach the same answer. Policy iteration needed only a")
print("handful of iterations because each one solves the evaluation exactly;")
print("value iteration needed 22 smaller steps to creep to the same values.")
