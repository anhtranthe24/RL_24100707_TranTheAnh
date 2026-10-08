import gymnasium as gym
import numpy as np
from collections import defaultdict
from mc_utils import generate_episode, compute_returns

def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0
    return 1

def every_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for _ in range(n_episodes):
        episode = generate_episode(env, policy)
        rewards = [step[2] for step in episode]
        episode_returns = compute_returns(rewards, gamma)

        for (state, action, reward), G in zip(episode, episode_returns):
            returns_sum[state] += G
            returns_count[state] += 1

    V = {
        state: returns_sum[state] / returns_count[state]
        for state in returns_count
    }

    return V, returns_count

env = gym.make("Blackjack-v1")

V, returns_count = every_visit_mc_prediction(
    env,
    stick_on_20_policy,
    n_episodes=10000,
    gamma=1.0
)
print("Number of estimated  (Số lượng trạng thái ước tính):", len(V))
for state, value in list(V.items())[:10]:
    print(
        f"State (Trạng thái): {state}, "
        f"V(s): {value:.4f}, "
        f"Count (Đếm): {returns_count[state]}"
    )

env.close()