import gymnasium as gym
import numpy as np
from mc_utils import generate_episode, compute_returns
def random_policy(state):
    return np.random.randint(0, 2)
env=gym.make("Blackjack-v1")
episode = generate_episode(env, random_policy)
rewards = [step[2] for step in episode]
returns = compute_returns(rewards, gamma=1.0)
trajectory = [
    (state, action, reward, G) for (state, action, reward), G in zip(episode, returns)
]
for i, (state, action, reward, G) in enumerate(trajectory):
    print(f"Step {i + 1}")
    print("State:", state)
    print("Action: ", action)
    print("Reward:", reward)
    print("Return:", G)
env.close()