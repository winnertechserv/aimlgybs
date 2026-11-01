"""
Chapter 33, Example 2: Q-Learning on the Same Grid World
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
Value iteration needed the transition model; Q-learning does not. The agent learns
purely from experience using the temporal-difference update and an epsilon-greedy policy
that explores early and exploits later. After 8000 episodes it recovers the SAME optimal
policy and approaches the same state values, without ever being told the transition
probabilities.

Run it:           python ex2_q_learning_on_the_same_grid_world.py
Expected output:  expected_output/ex2_q_learning_on_the_same_grid_world.txt
"""
# Chapter 33 - Example 2: Q-Learning on the same 4x3 Grid World
import numpy as np
np.random.seed(0)

print("=== Q-Learning: Model-Free Reinforcement Learning ===\n")
print("Value iteration NEEDS the transition model P(s'|s,a) and reward function.")
print("Q-Learning does NOT: the agent learns purely from experience by")
print("interacting with the environment.\n")
print("Update rule (temporal-difference):")
print("  Q(s,a) <- Q(s,a) + alpha [ r + gamma*max_a' Q(s',a') - Q(s,a) ]")
print("            \\____________________ TD error ____________________/\n")

rows, cols = 3, 4
WALL      = (2, 2)
TERMINALS = {(4, 3): +1.0, (4, 2): -1.0}
R_STEP, GAMMA, ALPHA = -0.04, 1.0, 0.1
ACTIONS = ["Up", "Down", "Left", "Right"]
DELTA   = {"Up": (0, 1), "Down": (0, -1), "Left": (-1, 0), "Right": (1, 0)}
PERP    = {"Up": ["Left","Right"], "Down": ["Left","Right"],
           "Left": ["Up","Down"],  "Right": ["Up","Down"]}
states  = [(c, r) for c in range(1, cols+1) for r in range(1, rows+1) if (c, r) != WALL]

def move(s, a):
    dc, dr = DELTA[a]; ns = (s[0]+dc, s[1]+dr)
    return s if (ns == WALL or not (1 <= ns[0] <= cols and 1 <= ns[1] <= rows)) else ns

def step(s, a):
    """The ENVIRONMENT: stochastic outcome, agent does not know these probabilities."""
    u = np.random.rand()
    actual = a if u < 0.8 else (PERP[a][0] if u < 0.9 else PERP[a][1])
    ns = move(s, actual)
    r  = TERMINALS[ns] if ns in TERMINALS else R_STEP
    return ns, r, ns in TERMINALS

Q = {(s, a): 0.0 for s in states for a in ACTIONS}

def eps_greedy(s, eps):
    if np.random.rand() < eps:
        return ACTIONS[np.random.randint(4)]          # EXPLORE
    qs = [Q[(s, a)] for a in ACTIONS]
    return ACTIONS[int(np.argmax(qs))]                 # EXPLOIT

EPISODES = 8000
print(f"Training for {EPISODES} episodes with epsilon-greedy exploration")
print("(epsilon decays from 1.0 to 0.05: explore early, exploit later)\n")
print(f"{'episode':>9}{'epsilon':>10}{'Q(1,1) best':>14}{'avg return':>13}")
print("-" * 48)

returns = []
for ep in range(1, EPISODES + 1):
    eps = max(0.05, 1.0 - ep / (EPISODES * 0.6))
    s, total, done, steps = (1, 1), 0.0, False, 0
    while not done and steps < 100:
        a = eps_greedy(s, eps)
        ns, r, done = step(s, a)
        best_next = 0.0 if done else max(Q[(ns, a2)] for a2 in ACTIONS)
        Q[(s, a)] += ALPHA * (r + GAMMA * best_next - Q[(s, a)])
        s = ns; total += r; steps += 1
    returns.append(total)
    if ep % 1600 == 0:
        print(f"{ep:>9}{eps:>10.3f}{max(Q[((1,1),a)] for a in ACTIONS):>14.4f}"
              f"{np.mean(returns[-500:]):>13.4f}")

print("\n=== Learned Q-values for the START state (1,1) ===\n")
for a in ACTIONS:
    bar = "#" * int(max(0, Q[((1,1), a)]) * 40)
    print(f"  Q((1,1), {a:<5}) = {Q[((1,1),a)]:+.4f}  {bar}")
best_a = max(ACTIONS, key=lambda a: Q[((1,1), a)])
print(f"  -> best action from START: {best_a}")

print("\n=== Learned policy (from Q-values only, no model used) ===\n")
ARROW = {"Up":"^","Down":"v","Left":"<","Right":">"}
for r in range(rows, 0, -1):
    line = f"  row {r} |"
    for c in range(1, cols+1):
        s = (c, r)
        if s == WALL:            line += "   #  "
        elif s in TERMINALS:     line += f"  {'+1' if TERMINALS[s]>0 else '-1':>2}  "
        else:
            a = max(ACTIONS, key=lambda a: Q[(s, a)])
            line += f"   {ARROW[a]}  "
    print(line)
print("          " + "".join(f"{'col '+str(c):>6}" for c in range(1, cols+1)))

print("\n=== V(s) = max_a Q(s,a)  compared with value iteration ===\n")
vi = {(1,1):0.7053, (1,3):0.8116, (3,3):0.9178, (4,1):0.3878}
print(f"  {'state':<8}{'Q-learning':>12}{'value iteration':>18}{'difference':>13}")
print("  " + "-" * 52)
for s, u in vi.items():
    v = max(Q[(s, a)] for a in ACTIONS)
    print(f"  {str(s):<8}{v:>12.4f}{u:>18.4f}{abs(v-u):>13.4f}")

print("\nQ-learning approaches the same values WITHOUT ever being told the")
print("transition probabilities - it learned them implicitly from experience.")
