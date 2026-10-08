import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mc_utils import generate_episode, compute_returns

def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0
    return 1

def update_first_visit(returns_sum, returns_count, episode, gamma):
    rewards = [step[2] for step in episode]
    episode_returns = compute_returns(rewards, gamma)
    visited = set()

    for (state, action, reward), G in zip(episode, episode_returns):
        if state in visited:
            continue

        visited.add(state)
        returns_sum[state] += G
        returns_count[state] += 1

def update_every_visit(returns_sum, returns_count, episode, gamma):
    rewards = [step[2] for step in episode]
    episode_returns = compute_returns(rewards, gamma)

    for (state, action, reward), G in zip(episode, episode_returns):
        returns_sum[state] += G
        returns_count[state] += 1

env = gym.make("Blackjack-v1")

target_states = [
    (20, 10, False),
    (18, 6, False),
    (13, 2, False)
]

checkpoints = [100, 500, 1000, 5000, 10000]
gamma = 1.0
seed = 42

first_sum = defaultdict(float)
first_count = defaultdict(int)
every_sum = defaultdict(float)
every_count = defaultdict(int)

first_history = {state: [] for state in target_states}
every_history = {state: [] for state in target_states}

for episode_number in range(1, max(checkpoints) + 1):
    episode = generate_episode(
        env,
        stick_on_20_policy,
        seed=seed + episode_number
    )

    update_first_visit(
        first_sum,
        first_count,
        episode,
        gamma
    )

    update_every_visit(
        every_sum,
        every_count,
        episode,
        gamma
    )

    if episode_number in checkpoints:
        for state in target_states:
            if state in first_count:
                first_value = first_sum[state] / first_count[state]
            else:
                first_value = np.nan

            if state in every_count:
                every_value = every_sum[state] / every_count[state]
            else:
                every_value = np.nan

            first_history[state].append(first_value)
            every_history[state].append(every_value)

        print(f"Episodes: {episode_number}")

        for state in target_states:
            print(
                f"{state} | "
                f"First-Visit: {first_history[state][-1]:.4f} | "
                f"Every-Visit: {every_history[state][-1]:.4f}"
            )

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

plt.figure(figsize=(10, 6))

for state in target_states:
    plt.plot(
        checkpoints,
        first_history[state],
        marker="o",
        label=f"First-Visit {state}"
    )
    plt.plot(
        checkpoints,
        every_history[state],
        marker="x",
        linestyle="--",
        label=f"Every-Visit {state}"
    )

plt.xlabel("Number of Episodes")
plt.ylabel("V(s)")
plt.title("First-Visit vs Every-Visit Monte Carlo")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(figures_dir, "first_vs_every_visit.png"),
    dpi=300
)

plt.show()
env.close()