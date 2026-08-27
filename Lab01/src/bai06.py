import gymnasium as gym
import numpy as np
env = gym.make("CartPole-v1")
actions = [env.action_space.sample() for _ in range(20)]
print(f"Danh sách 20 actions:\n{actions}")
unique_actions, counts = np.unique(actions, return_counts=True)
frequencies = dict(zip(unique_actions, counts))
print(f"Tần suất xuất hiện từng action:\n{frequencies}")
env.close()