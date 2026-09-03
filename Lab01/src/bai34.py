import numpy as np
from bai33 import run_episode
import gymnasium as gym
def evaluate_policy(env_name, policy, n_episodes=100, seed=42):
    env = gym.make(env_name)
    rewards, lengths = [], []
    for i in range(n_episodes):
        res = run_episode(env, policy, seed=seed+i)
        rewards.append(res["reward"])
        lengths.append(res["length"])
    env.close()
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_length": np.mean(lengths),
        "all_rewards": rewards
    }