import gymnasium as gym
import numpy as np
from collections import defaultdict

env = gym.make("Blackjack-v1")

Q = defaultdict(
    lambda: np.zeros(env.action_space.n)
)

returns_sum = defaultdict(
    lambda: np.zeros(env.action_space.n)
)

returns_count = defaultdict(
    lambda: np.zeros(env.action_space.n, dtype=int)
)

test_states = [
    (20, 10, False),
    (18, 6, False),
    (13, 2, False)
]

for state in test_states:
    print(f"State: {state}")
    print(f"Q: {Q[state]}")
    print(f"Returns Sum: {returns_sum[state]}")
    print(f"Returns Count: {returns_count[state]}")

env.close()