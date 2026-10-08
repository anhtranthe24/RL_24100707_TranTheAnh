import gymnasium as gym
import numpy as np
from collections import defaultdict

def epsilon_greedy_action(Q, state, n_actions, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))

    return int(np.argmax(Q[state]))

def compute_returns(rewards, gamma=1.0):
    returns = [0.0] * len(rewards)
    G = 0.0

    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns

def on_policy_mc_control(
    env,
    n_episodes,
    gamma=1.0,
    epsilon=0.1,
    seed=42
):
    Q = defaultdict(
        lambda: np.zeros(env.action_space.n)
    )

    N = defaultdict(
        lambda: np.zeros(env.action_space.n, dtype=int)
    )

    rng = np.random.default_rng(seed)
    episode_rewards = []

    for episode_number in range(n_episodes):
        state, info = env.reset(
            seed=seed + episode_number
        )

        episode = []
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = epsilon_greedy_action(
                Q,
                state,
                env.action_space.n,
                epsilon,
                rng
            )

            next_state, reward, terminated, truncated, info = env.step(action)

            episode.append((state, action, reward))
            state = next_state

        rewards = [step[2] for step in episode]
        returns = compute_returns(rewards, gamma)

        episode_rewards.append(sum(rewards))

        visited = set()

        for (state, action, reward), G in zip(episode, returns):
            state_action = (state, action)

            if state_action in visited:
                continue

            visited.add(state_action)

            N[state][action] += 1
            Q[state][action] += (
                G - Q[state][action]
            ) / N[state][action]

    policy = {}

    for state in Q:
        policy[state] = int(np.argmax(Q[state]))

    return Q, policy, episode_rewards

def main():
    env = gym.make("Blackjack-v1")

    Q, policy, episode_rewards = on_policy_mc_control(
        env,
        n_episodes=10000,
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

    for state in test_states:
        print(
            f"State: {state} | "
            f"Q: {Q[state]} | "
            f"Policy: {policy[state]}"
        )

    print("Total episodes:", len(episode_rewards))

    env.close()

if __name__ == "__main__":
    main()