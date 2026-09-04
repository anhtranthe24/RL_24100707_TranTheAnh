import numpy as np

def q_from_v(env, V, state, action, gamma=0.99):
    q = 0.0
    for p, next_state, reward, terminated in env.unwrapped.P[state][action]:
        q += p * (reward + gamma * (0 if terminated else V[next_state]))
    return q

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    V = np.zeros(env.observation_space.n)
    for i in range(max_iterations):
        new_V = np.zeros_like(V)
        for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                new_V[state] += policy[state][action] * q_from_v(env, V, state, action, gamma)
        delta = np.max(np.abs(new_V - V))
        V = new_V
        if delta < theta:
            return V, i + 1
    return V, max_iterations

def greedy_policy_from_value(env, V, gamma=0.99):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        q_values = [q_from_v(env, V, state, action, gamma) for action in range(env.action_space.n)]
        policy[state] = np.argmax(q_values)
    return policy

def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for i in range(max_iterations):
        V = np.zeros(env.observation_space.n)
        for _ in range(max_iterations):
            new_V = np.array([
                q_from_v(env, V, state, policy[state], gamma)
                for state in range(env.observation_space.n)
            ])
            delta = np.max(np.abs(new_V - V))
            V = new_V
            if delta < theta:
                break
        new_policy = greedy_policy_from_value(env, V, gamma)
        if np.array_equal(policy, new_policy):
            return new_policy, V, i + 1
        policy = new_policy
    return policy, V, max_iterations

def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    V = np.zeros(env.observation_space.n)
    deltas = []
    for i in range(max_iterations):
        new_V = np.zeros_like(V)
        for state in range(env.observation_space.n):
            q_values = [q_from_v(env, V, state, action, gamma) for action in range(env.action_space.n)]
            new_V[state] = np.max(q_values)
        delta = np.max(np.abs(new_V - V))
        deltas.append(delta)
        V = new_V
        if delta < theta:
            return V, i + 1, deltas
    return V, max_iterations, deltas

def evaluate_policy_by_simulation(env, policy, n_episodes=1000, seed=42):
    rewards = []
    lengths = []
    successes = 0
    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0
        length = 0
        while True:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            length += 1
            if terminated or truncated:
                break
        rewards.append(total_reward)
        lengths.append(length)
        if total_reward > 0:
            successes += 1
    return {
        "success_rate": successes / n_episodes,
        "mean_reward": np.mean(rewards),
        "mean_length": np.mean(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths)
    }

def print_frozenlake_policy(env, policy):
    symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    desc = env.unwrapped.desc
    n = desc.shape[0]
    for r in range(n):
        row = []
        for c in range(n):
            state = r * n + c
            cell = desc[r][c].decode()
            row.append(cell if cell in ["H", "G"] else symbols[policy[state]])
        print(" ".join(row))