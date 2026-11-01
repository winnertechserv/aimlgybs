"""
Chapter 33, Example 1: Value Iteration on the 4x3 Grid World
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This solves the classic 4x3 grid world: wall at (2,2), terminal states +1 and -1, step
reward -0.04, and actions succeeding with probability 0.8 while slipping sideways with
probability 0.1 each. Applying the Bellman update repeatedly converges in 22 iterations
to the published utilities - U(1,1)=0.705, U(1,3)=0.812, U(3,3)=0.918 - and the
extracted policy routes the agent safely around the wall.

Run it:           python ex1_value_iteration_on_the_4x3_grid_world.py
Expected output:  expected_output/ex1_value_iteration_on_the_4x3_grid_world.txt
"""
# Chapter 33 - Example 1: Value Iteration on the 4x3 Grid World
import numpy as np
np.set_printoptions(suppress=True)

print("=== Value Iteration on the 4x3 Grid World ===\n")
print("Grid (column, row), (1,1) is bottom-left:\n")
print("     3 | . . . +1")
print("     2 | . # . -1        # = wall/forbidden square (2,2)")
print("     1 | S . . .         S = START")
print("       +---------")
print("       1 2 3 4\n")
print("Stochastic environment: the intended action succeeds with p=0.8,")
print("and moves at right angles with p=0.1 each. Bumping into a wall")
print("leaves the agent in the same square.")
print("Reward: -0.04 for every non-terminal transition. Discount gamma=1.0\n")

rows, cols = 3, 4
WALL      = (2, 2)
TERMINALS = {(4, 3): +1.0, (4, 2): -1.0}
R_STEP    = -0.04
GAMMA     = 1.0

states = [(c, r) for c in range(1, cols+1) for r in range(1, rows+1)
          if (c, r) != WALL]
ACTIONS = {"Up": (0, 1), "Down": (0, -1), "Left": (-1, 0), "Right": (1, 0)}
RIGHT_ANGLES = {"Up": ["Left", "Right"], "Down": ["Left", "Right"],
                "Left": ["Up", "Down"],  "Right": ["Up", "Down"]}

def move(s, a):
    dc, dr = ACTIONS[a]
    ns = (s[0] + dc, s[1] + dr)
    if ns == WALL or not (1 <= ns[0] <= cols and 1 <= ns[1] <= rows):
        return s                      # bumped into a wall -> stay
    return ns

def transitions(s, a):
    """Returns list of (probability, next_state)."""
    out = [(0.8, move(s, a))]
    for ra in RIGHT_ANGLES[a]:
        out.append((0.1, move(s, ra)))
    return out

# ---- Value iteration ----
U = {s: 0.0 for s in states}
for s, v in TERMINALS.items():
    U[s] = v

print("Bellman update:  U(s) <- max_a  sum_s' P(s'|s,a) [ R(s,a,s') + gamma*U(s') ]\n")
print(f"{'iter':>5}{'U(1,1)':>10}{'U(1,3)':>10}{'U(3,3)':>10}{'U(4,1)':>10}{'max change':>13}")
print("-" * 58)

for it in range(1, 60):
    U_new = dict(U)
    delta = 0.0
    for s in states:
        if s in TERMINALS:
            continue
        best = -1e9
        for a in ACTIONS:
            q = sum(p * (R_STEP + GAMMA * U[ns]) for p, ns in transitions(s, a))
            best = max(best, q)
        U_new[s] = best
        delta = max(delta, abs(best - U[s]))
    U = U_new
    if it <= 5 or it % 10 == 0 or delta < 1e-4:
        print(f"{it:>5}{U[(1,1)]:>10.4f}{U[(1,3)]:>10.4f}{U[(3,3)]:>10.4f}{U[(4,1)]:>10.4f}{delta:>13.6f}")
    if delta < 1e-4:
        print(f"\nConverged after {it} iterations (max change < 1e-4)")
        break

# ---- Final utilities ----
print("\n=== Converged utility values U(s) ===\n")
for r in range(rows, 0, -1):
    line = f"  row {r} |"
    for c in range(1, cols+1):
        line += "   WALL " if (c, r) == WALL else f"{U[(c,r)]:>8.4f}"
    print(line)
print("          " + "".join(f"{'col '+str(c):>8}" for c in range(1, cols+1)))

# ---- Extract the optimal policy ----
print("\n=== Optimal policy (best action in every state) ===\n")
ARROW = {"Up": "^", "Down": "v", "Left": "<", "Right": ">"}
policy = {}
for s in states:
    if s in TERMINALS: continue
    qs = {a: sum(p * (R_STEP + GAMMA * U[ns]) for p, ns in transitions(s, a))
          for a in ACTIONS}
    policy[s] = max(qs, key=qs.get)

for r in range(rows, 0, -1):
    line = f"  row {r} |"
    for c in range(1, cols+1):
        if (c, r) == WALL:              line += "   #  "
        elif (c, r) in TERMINALS:       line += f"  {'+1' if TERMINALS[(c,r)]>0 else '-1':>2}  "
        else:                           line += f"   {ARROW[policy[(c,r)]]}  "
    print(line)
print("          " + "".join(f"{'col '+str(c):>6}" for c in range(1, cols+1)))

print("\nThe policy routes the agent up and around the wall to reach +1 while")
print("avoiding the -1 square, just as expected.")
print(f"\nUtility of the START state (1,1): {U[(1,1)]:.4f}")
print("Sanity check: reaching +1 in 5 steps gives 4*(-0.04) + 1 = 0.84,")
print("and the converged utility is lower because of the stochastic slips.")
