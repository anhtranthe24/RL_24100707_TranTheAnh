import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from mc_utils import generate_episode, compute_returns
from bai32 import on_policy_mc_control


def random_policy(state, rng):
    return int(rng.integers(2))


def fixed_policy(state):
    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:
        return 0

    return 1


def epsilon_greedy_action(Q, state, n_actions, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))

    return int(np.argmax(Q[state]))


def first_visit_mc_prediction(env, policy, n_episodes, gamma=1.0, seed=42):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for episode_number in range(n_episodes):
        episode = generate_episode(
            env,
            policy,
            seed=seed + episode_number
        )

        rewards = [step[2] for step in episode]
        episode_returns = compute_returns(rewards, gamma)
        visited = set()

        for (state, action, reward), G in zip(
            episode,
            episode_returns
        ):
            if state in visited:
                continue

            visited.add(state)
            returns_sum[state] += G
            returns_count[state] += 1

    V = {
        state: returns_sum[state] / returns_count[state]
        for state in returns_sum
    }

    return V


def every_visit_mc_prediction(env, policy, n_episodes, gamma=1.0, seed=42):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for episode_number in range(n_episodes):
        episode = generate_episode(
            env,
            policy,
            seed=seed + episode_number
        )

        rewards = [step[2] for step in episode]
        episode_returns = compute_returns(rewards, gamma)

        for (state, action, reward), G in zip(
            episode,
            episode_returns
        ):
            returns_sum[state] += G
            returns_count[state] += 1

    V = {
        state: returns_sum[state] / returns_count[state]
        for state in returns_sum
    }

    return V


def evaluate_policy(env, policy, n_episodes=10000, seed=123):
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0

    rng = np.random.default_rng(seed)

    for episode_number in range(n_episodes):
        state, info = env.reset(seed=seed + episode_number)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = policy(state, rng)
            state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    return {
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "mean_reward": total_reward / n_episodes
    }


def evaluate_fixed_policy(env, policy, n_episodes=10000, seed=123):
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0

    for episode_number in range(n_episodes):
        state, info = env.reset(seed=seed + episode_number)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = policy(state)
            state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    return {
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "mean_reward": total_reward / n_episodes
    }


def main():
    env = gym.make("Blackjack-v1")

    print("Environment")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    print("\nSample Episode")

    episode = generate_episode(
        env,
        fixed_policy,
        seed=42
    )

    print("Trajectory length:", len(episode))

    for step in episode:
        print(
            f"State: {step[0]} | "
            f"Action: {step[1]} | "
            f"Reward: {step[2]}"
        )

    rewards = [step[2] for step in episode]
    returns = compute_returns(rewards, gamma=1.0)
    discounted_returns = compute_returns(rewards, gamma=0.9)

    print("\nReturns")
    print("Rewards:", rewards)
    print("Gamma = 1.0:", returns)
    print("Gamma = 0.9:", discounted_returns)

    print("\nFirst-Visit MC Prediction")

    V_first = first_visit_mc_prediction(
        env,
        fixed_policy,
        n_episodes=10000,
        gamma=1.0
    )

    print("Number of states:", len(V_first))

    print("\nEvery-Visit MC Prediction")

    V_every = every_visit_mc_prediction(
        env,
        fixed_policy,
        n_episodes=10000,
        gamma=1.0
    )

    print("Number of states:", len(V_every))

    common_states = set(V_first) & set(V_every)

    if common_states:
        differences = [
            abs(V_first[state] - V_every[state])
            for state in common_states
        ]

        print(
            "Mean absolute difference:",
            np.mean(differences)
        )

    print("\nMonte Carlo Control")

    Q, learned_policy, episode_rewards = on_policy_mc_control(
        env,
        n_episodes=100000,
        gamma=1.0,
        epsilon=0.1,
        seed=42
    )

    print("Number of learned states:", len(Q))

    test_states = [
        (20, 10, False),
        (18, 6, False),
        (13, 2, False),
        (18, 6, True)
    ]

    print("\nLearned Policy")

    for state in test_states:
        if state in Q:
            action = int(np.argmax(Q[state]))
            action_name = "Stick" if action == 0 else "Hit"

            print(
                f"State: {state} | "
                f"Q: {Q[state]} | "
                f"Action: {action} ({action_name})"
            )
        else:
            print(f"State: {state} | Not visited")

    print("\nPolicy Evaluation")

    random_result = evaluate_policy(
        env,
        random_policy,
        n_episodes=10000,
        seed=123
    )

    fixed_result = evaluate_fixed_policy(
        env,
        fixed_policy,
        n_episodes=10000,
        seed=123
    )

    def learned_policy_function(state, rng):
        if state in Q:
            return int(np.argmax(Q[state]))

        return int(rng.integers(env.action_space.n))

    learned_result = evaluate_policy(
        env,
        learned_policy_function,
        n_episodes=10000,
        seed=123
    )

    results = {
        "Random": random_result,
        "Fixed": fixed_result,
        "Learned": learned_result
    }

    for name, result in results.items():
        print(f"\n{name} Policy")
        print(f"Win rate: {result['win_rate']:.4f}")
        print(f"Loss rate: {result['loss_rate']:.4f}")
        print(f"Draw rate: {result['draw_rate']:.4f}")
        print(f"Mean reward: {result['mean_reward']:.4f}")

    rewards = np.asarray(episode_rewards, dtype=float)
    window = 1000

    moving_average = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        np.arange(window, len(rewards) + 1),
        moving_average
    )

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Monte Carlo Control Learning Curve")
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(
        figures_dir,
        "mc_convergence.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.show()

    print("\nLearning curve saved to:", output_path)

    env.close()


if __name__ == "__main__":
    main()