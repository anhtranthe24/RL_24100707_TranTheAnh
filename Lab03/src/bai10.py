import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mc_utils import generate_episode, compute_returns

def random_policy(state):
    return np.random.randint(0, 2)

env = gym.make("Blackjack-v1")
gammas = [0.5, 0.8, 0.9, 0.99, 1.0]
n_episodes = 10000
mean_returns = []

for gamma in gammas:
    initial_returns = []
    for _ in range(n_episodes):
        episode = generate_episode(env, random_policy)
        rewards = [step[2] for step in episode]
        returns = compute_returns(rewards, gamma)
        initial_returns.append(returns[0])
    mean_returns.append(np.mean(initial_returns))

for gamma, mean_return in zip(gammas, mean_returns):
    print(f"Gamma = {gamma}: Mean G0 = {mean_return:.4f}")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(gammas, mean_returns, marker="o")
plt.xlabel("Gamma")
plt.ylabel("Mean G0")
plt.title("Mean Initial Return for Different Gamma Values")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "gamma_comparison.png"), dpi=300)
plt.show()

env.close()