"""
Chapter 34, Example 2: SARSA vs Q-Learning on Cliff Walking
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
This is the classic example that shows why on-policy and off-policy learning behave
differently. Both agents learn to cross a grid with a cliff along the bottom, while
still exploring randomly 10% of the time. Q-learning finds the shortest route, right
along the cliff edge. SARSA finds a longer route along the top. Now check the reward
table: SARSA earns about -27 per episode against Q-learning's -52, because Q-learning's
random steps keep pushing it off the cliff. Shortest is not the same as safest.

Run it:           python ex2_sarsa_vs_q_learning_on_cliff_walking.py
Expected output:  expected_output/ex2_sarsa_vs_q_learning_on_cliff_walking.txt
"""
# Chapter 34 - Example 2: SARSA (on-policy) vs Q-learning (off-policy) on Cliff Walking
import numpy as np

print("=== SARSA vs Q-Learning: The Cliff-Walking Problem ===\n")
print("A 4x12 grid. Start S is bottom-left, goal G is bottom-right.")
print("The cells between them on the bottom row are a CLIFF.")
print("Every step costs -1. Stepping into the cliff costs -100 and sends you")
print("back to S. Both agents explore with epsilon-greedy (eps = 0.1).\n")
print("    . . . . . . . . . . . .")
print("    . . . . . . . . . . . .")
print("    . . . . . . . . . . . .")
print("    S C C C C C C C C C C G      C = cliff\n")

ROWS, COLS = 4, 12
START, GOAL = (3, 0), (3, 11)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # up, down, left, right
NAMES = ["^", "v", "<", ">"]
EPS, ALPHA, GAMMA, EPISODES, RUNS = 0.1, 0.5, 1.0, 500, 20

def step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    r, c = min(max(r, 0), ROWS - 1), min(max(c, 0), COLS - 1)
    # Falling sends you back to S and the episode continues (as in Sutton & Barto's
    # original version); some libraries end the episode here instead.
    if r == 3 and 1 <= c <= 10:                      # fell off the cliff
        return START, -100, False
    return (r, c), -1, (r, c) == GOAL

def choose(Q, s, rng):
    if rng.random() < EPS:
        return int(rng.integers(4))                  # explore
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))   # exploit (break ties randomly)

def train(method, seed):
    rng = np.random.default_rng(seed)
    Q = np.zeros((ROWS, COLS, 4))
    returns = []
    for _ in range(EPISODES):
        s, total = START, 0
        a = choose(Q, s, rng)
        for _ in range(500):
            ns, r, done = step(s, a)
            total += r
            na = choose(Q, ns, rng)
            if method == "sarsa":
                target = r + (0 if done else GAMMA * Q[ns][na])        # the action it WILL take
            else:
                target = r + (0 if done else GAMMA * Q[ns].max())      # the BEST action
            Q[s][a] += ALPHA * (target - Q[s][a])
            s, a = ns, na
            if done:
                break
        returns.append(total)
    return Q, np.array(returns)

def greedy_path(Q):
    s, path = START, [START]
    for _ in range(40):
        s, _, done = step(s, int(np.argmax(Q[s])))
        path.append(s)
        if done:
            break
    return path

def draw(path):
    grid = [["." for _ in range(COLS)] for _ in range(ROWS)]
    for c in range(1, 11):
        grid[3][c] = "C"
    for (r, c) in path:
        grid[r][c] = "*"
    grid[3][0], grid[3][11] = "S", "G"
    for row in grid:
        print("    " + " ".join(row))

results = {}
for method in ["sarsa", "qlearning"]:
    all_returns = []
    for seed in range(RUNS):
        Q, ret = train(method, seed)
        all_returns.append(ret)
    results[method] = (Q, np.mean(all_returns, axis=0))

print(f"Average reward per episode during training ({RUNS} runs averaged):\n")
print(f"  {'episodes':<12}{'SARSA':>10}{'Q-learning':>14}")
print("  " + "-" * 36)
for lo, hi in [(0, 50), (50, 100), (100, 200), (200, 350), (350, 500)]:
    s = results["sarsa"][1][lo:hi].mean()
    q = results["qlearning"][1][lo:hi].mean()
    print(f"  {lo+1:>3}-{hi:<8}{s:>10.1f}{q:>14.1f}")

for method, label in [("sarsa", "SARSA"), ("qlearning", "Q-learning")]:
    path = greedy_path(results[method][0])
    print(f"\n{label} learned path ({len(path) - 1} steps):")
    draw(path)

print("\nWhat this shows:")
print("  Q-learning learns the SHORTEST path, right along the cliff edge. Its")
print("  update uses max Q(s',a'), the value of the best action, as if it will")
print("  never explore again. That is OFF-policy learning.")
print("  SARSA learns a SAFER path that keeps well away from the edge. Its update uses the")
print("  action it will actually take next, random exploration included, so it")
print("  knows a random step near the edge can cost -100. That is ON-policy.")
print("  So while both are still exploring, SARSA earns MORE reward per episode")
print("  even though Q-learning's final route is shorter.")
