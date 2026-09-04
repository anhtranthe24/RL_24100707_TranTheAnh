import gymnasium as gym
import numpy as np
from mdp_utils import q_from_v, greedy_policy_from_value
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
old_policy = np.zeros(env.observation_space.n, dtype=int)
V = np.zeros(env.observation_space.n)
for _ in range(10000):
    new_V = np.array([q_from_v(env, V, s, old_policy[s], 0.99) for s in range(env.observation_space.n)])
    if np.max(np.abs(new_V - V)) < 1e-8:
        V = new_V
        break
    V = new_V
new_policy = greedy_policy_from_value(env, V)
changed = np.sum(old_policy != new_policy)
print("Chính sách cũ:", old_policy)
print("Chính sách mới:", new_policy)
print("Số trạng thái thay đổi:", changed)
env.close()