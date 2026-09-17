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
env = gym.make("Blackjack-v1")
target_state = (20, 10, False)
checkpoints = [100, 500, 1000, 5000, 10000]
returns = defaultdict(list)
values = []
for episode_number in range(1, max(checkpoints) + 1):
    episode = generate_episode(env, stick_on_20_policy)
    rewards = [step[2] for step in episode]
    episode_returns = compute_returns(rewards, gamma=1.0)
    for (state, action, reward), G in zip(episode, episode_returns):
        returns[state].append(G)
    if episode_number in checkpoints:
        if target_state in returns:
            value = np.mean(returns[target_state])
            values.append(value)
            print(f"Episodes = {episode_number}: V{target_state} = {value:.4f}")
        else:
            values.append(np.nan)
            print(f"Episodes = {episode_number}: V{target_state} = N/A")
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(checkpoints, values, marker="o")
plt.xlabel("Number of Episodes")
plt.ylabel(f"V{target_state}")
plt.title("Monte Carlo Value Convergence")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "mc_convergence.png"), dpi=300)
plt.show()
env.close()