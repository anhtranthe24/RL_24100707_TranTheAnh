import gymnasium as gym
import numpy as np
from bai21 import (
    stick_on_20_policy,
    first_visit_from_episodes,
    every_visit_from_episodes
)
from mc_utils import generate_episode

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

common_states = set(V_first) & set(V_every)

differences = {
    state: abs(V_first[state] - V_every[state])
    for state in common_states
}

mean_difference = np.mean(list(differences.values()))
max_state = max(differences, key=differences.get)
max_difference = differences[max_state]

print("Number of common states (Số lượng trạng thái):", len(common_states))
print(f"Mean absolute differencem (Sự khác biệt tuyệt đối trung bình): {mean_difference:.6f}")
print(f"Maximum difference (Chênh lệch tối đa): {max_difference:.6f}")
print(f"State with maximum difference (Trạng thái có chênh lệch tối đa): {max_state}")

print("\nTop 10 differences:")

for state, difference in sorted(
    differences.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]:
    print(
        f"{state}: "
        f"First={V_first[state]:.4f}, "
        f"Every={V_every[state]:.4f}, "
        f"Difference={difference:.4f}"
    )

env.close()