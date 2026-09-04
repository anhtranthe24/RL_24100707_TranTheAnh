import gymnasium as gym
import numpy as np
from mdp_utils import value_iteration, policy_iteration, greedy_policy_from_value, evaluate_policy_by_simulation
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
rng = np.random.default_rng(42)
random_policy = rng.integers(0, env.action_space.n, size=env.observation_space.n)
V, _, _ = value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000)
vi_policy = greedy_policy_from_value(env, V, gamma=0.99)
pi_policy, _, _ = policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000)
for name, policy in [
    ("Random", random_policy),
    ("Value Iteration", vi_policy),
    ("Policy Iteration", pi_policy)
]:
    result = evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42)
    print(f"\n{name}")
    print("Success rate:", result["success_rate"])
    print("Mean reward:", result["mean_reward"])
    print("Mean episode length:", result["mean_length"])
    print("Min episode length:", result["min_length"])
    print("Max episode length:", result["max_length"])
env.close()