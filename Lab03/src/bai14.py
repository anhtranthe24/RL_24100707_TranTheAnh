import gymnasium as gym
import numpy as np
from collections import defaultdict
from mc_utils import generate_episode, compute_returns
def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0
    return 1
env = gym.make("Blackjack-v1")
returns = defaultdict(list)
n_episodes = 10000
for _ in range(n_episodes):
    episode = generate_episode(env, stick_on_20_policy)
    rewards = [step[2] for step in episode]
    episode_returns = compute_returns(rewards, gamma=1.0)
    for (state, action, reward), G in zip(episode, episode_returns):
        returns[state].append(G)
V = {state: np.mean(state_returns) for state, state_returns in returns.items()}
print("Number of states:", len(V))
print("\nEstimated V(s):")
for state, value in list(V.items())[:10]:
    print(f"{state}: {value:.4f}")
env.close()