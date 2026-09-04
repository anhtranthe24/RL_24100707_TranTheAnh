import gymnasium as gym
from mdp_utils import policy_iteration
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
policy, V, n_iterations = policy_iteration(env)
print("Chính sách:", policy)
print("Policy Iteration hội tụ sau", n_iterations, "lần lặp.")
env.close()