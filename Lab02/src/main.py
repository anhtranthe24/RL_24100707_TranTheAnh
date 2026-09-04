"""
Lab02 Mini Project: MDP and Dynamic Programming on FrozenLake.
So sánh Value Iteration và Policy Iteration.
"""

import os
import csv
import gymnasium as gym
import matplotlib.pyplot as plt
from time import perf_counter
from mdp_utils import (
    q_from_v,
    policy_evaluation,
    greedy_policy_from_value,
    policy_iteration,
    value_iteration,
    evaluate_policy_by_simulation,
    print_frozenlake_policy
)

def create_environment(is_slippery=True):
    return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)

def get_transition_model(env):
    return env.unwrapped.P

def print_policy(env, policy):
    print_frozenlake_policy(env, policy)

def save_convergence_data(deltas):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "value_iteration_convergence.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "delta"])
        for i, delta in enumerate(deltas, 1):
            writer.writerow([i, delta])

def plot_convergence(deltas):
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, "value_iteration_convergence.png")
    plt.figure(figsize=(8, 5))
    plt.plot(deltas, label="Delta")
    plt.xlabel("Iteration")
    plt.ylabel("Delta")
    plt.title("Value Iteration Convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def main(is_slippery=True, gamma=0.99, theta=1e-8, max_iterations=10000):
    env = create_environment(is_slippery)
    P = get_transition_model(env)

    print("=== ENVIRONMENT ===")
    print("FrozenLake-v1 4x4")
    print("Slippery:", is_slippery)
    print("States:", env.observation_space.n)
    print("Actions:", env.action_space.n)
    print("Transition model:", len(P))

    start = perf_counter()
    vi_V, vi_iterations, vi_deltas = value_iteration(env, gamma, theta, max_iterations)
    vi_time = perf_counter() - start
    vi_policy = greedy_policy_from_value(env, vi_V, gamma)

    start = perf_counter()
    pi_policy, pi_V, pi_iterations = policy_iteration(env, gamma, theta, max_iterations)
    pi_time = perf_counter() - start

    vi_result = evaluate_policy_by_simulation(env, vi_policy, n_episodes=1000, seed=42)
    pi_result = evaluate_policy_by_simulation(env, pi_policy, n_episodes=1000, seed=42)

    print("\n=== VALUE ITERATION ===")
    print("Value table:")
    print(vi_V)
    print("Policy:")
    print_policy(env, vi_policy)
    print("Iterations:", vi_iterations)
    print("Runtime:", vi_time)
    print("Success rate:", vi_result["success_rate"])
    print("Mean reward:", vi_result["mean_reward"])
    print("Mean episode length:", vi_result["mean_length"])

    print("\n=== POLICY ITERATION ===")
    print("Value table:")
    print(pi_V)
    print("Policy:")
    print_policy(env, pi_policy)
    print("Iterations:", pi_iterations)
    print("Runtime:", pi_time)
    print("Success rate:", pi_result["success_rate"])
    print("Mean reward:", pi_result["mean_reward"])
    print("Mean episode length:", pi_result["mean_length"])

    print("\n=== COMPARISON ===")
    print(f"{'Algorithm':<20}{'Iterations':<12}{'Runtime':<15}{'Success Rate':<15}{'Mean Reward':<15}")
    print("-" * 77)
    print(f"{'Value Iteration':<20}{vi_iterations:<12}{vi_time:<15.6f}{vi_result['success_rate']:<15.4f}{vi_result['mean_reward']:<15.4f}")
    print(f"{'Policy Iteration':<20}{pi_iterations:<12}{pi_time:<15.6f}{pi_result['success_rate']:<15.4f}{pi_result['mean_reward']:<15.4f}")

    save_convergence_data(vi_deltas)
    plot_convergence(vi_deltas)

    env.close()

if __name__ == "__main__":
    for slippery in [False, True]:
        print("\n" + "=" * 60)
        print(f"RUN WITH slippery={slippery}")
        print("=" * 60)
        main(is_slippery=slippery)