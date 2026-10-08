import gymnasium as gym
import numpy as np
from collections import defaultdict
from mc_utils import generate_episode, compute_returns

def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0
    return 1

def first_visit_from_episodes(episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for episode in episodes:
        rewards = [step[2] for step in episode]
        episode_returns = compute_returns(rewards, gamma)
        visited = set()

        for (state, action, reward), G in zip(episode, episode_returns):
            if state in visited:
                continue

            visited.add(state)
            returns_sum[state] += G
            returns_count[state] += 1

    return {
        state: returns_sum[state] / returns_count[state]
        for state in returns_count
    }

def every_visit_from_episodes(episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for episode in episodes:
        rewards = [step[2] for step in episode]
        episode_returns = compute_returns(rewards, gamma)

        for (state, action, reward), G in zip(episode, episode_returns):
            returns_sum[state] += G
            returns_count[state] += 1

    return {
        state: returns_sum[state] / returns_count[state]
        for state in returns_count
    }

env = gym.make("Blackjack-v1")

n_episodes = 10000
gamma = 1.0
seed = 42

episodes = [
    generate_episode(env, stick_on_20_policy, seed=seed + i)
    for i in range(n_episodes)
]

V_first = first_visit_from_episodes(episodes, gamma)
V_every = every_visit_from_episodes(episodes, gamma)

common_states = list(set(V_first) & set(V_every))

print("Number of common states (Số lượng trạng thái):", len(common_states))
print("\nComparison (So sánh):")

for state in common_states[:10]:
    print(
        f"State (Trạng thái): {state} | "
        f"First-Visit (Ghé thăm lần đầu): {V_first[state]:.4f} | "
        f"Every-Visit (Mỗi lần ghé thăm): {V_every[state]:.4f}"
    )

env.close()