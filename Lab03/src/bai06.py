import gymnasium as gym
import numpy as np
from mc_utils import generate_episode
def random_policy(state):
    return np.random.randint(0,2)
env=gym.make("Blackjack-v1")
episode = generate_episode(env, random_policy)
rewards= [step[2] for step in episode]
total_reward = sum(rewards)
print("Episode length (Độ dài tập):", len(episode))
print("Rewards (Phần thưởng):", rewards)
print("Total reward (Tổng phần thưởng):", total_reward)
env.close()