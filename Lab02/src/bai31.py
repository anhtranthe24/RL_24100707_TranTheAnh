import gymnasium as gym
import numpy as np
from mdp_utils import q_from_v
def value_iteration_sweep(env, V, gamma):
    new_V = np.zeros_like(V)
    for state in range(env.observation_space.n):
        q_values = [q_from_v(env, V, state, action, gamma) for action in range(env.action_space.n)]
        new_V[state] = np.max(q_values)
    return new_V
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
V = np.zeros(env.observation_space.n)
new_V = value_iteration_sweep(env, V, 0.99)
print("V cũ:", V)
print("V mới:", new_V)
env.close()