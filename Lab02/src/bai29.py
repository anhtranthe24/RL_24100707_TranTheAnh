import gymnasium as gym
from mdp_utils import policy_iteration
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
policy, V, n_iterations = policy_iteration(env)
print("Chính sách tối ưu:", policy)
print("Giá trị V:", V)
print("Số lần lặp:", n_iterations)
env.close()