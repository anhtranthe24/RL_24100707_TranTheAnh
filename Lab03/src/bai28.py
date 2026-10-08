import numpy as np

def epsilon_greedy_action(Q_values, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(Q_values)))

    return int(np.argmax(Q_values))

Q_values = [1.0, 5.0]
epsilon = 0.1
n_trials = 10000
rng = np.random.default_rng(42)

action_counts = np.zeros(len(Q_values), dtype=int)

for _ in range(n_trials):
    action = epsilon_greedy_action(Q_values, epsilon, rng)
    action_counts[action] += 1

for action, count in enumerate(action_counts):
    frequency = count / n_trials
    print(
        f"Action {action}: "
        f"Count = {count}, "
        f"Frequency = {frequency:.4f}"
    )

greedy_action = int(np.argmax(Q_values))
greedy_frequency = action_counts[greedy_action] / n_trials

print(f"Greedy action: {greedy_action}")
print(f"Greedy action frequency: {greedy_frequency:.4f}")