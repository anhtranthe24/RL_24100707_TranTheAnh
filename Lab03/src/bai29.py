import os
import numpy as np
import matplotlib.pyplot as plt

def epsilon_greedy_action(Q_values, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(Q_values)))

    return int(np.argmax(Q_values))

Q_values = [1.0, 5.0]
epsilons = [0.01, 0.05, 0.10, 0.20, 0.50]
n_trials = 10000
rng = np.random.default_rng(42)

greedy_action = int(np.argmax(Q_values))
greedy_frequencies = []

for epsilon in epsilons:
    greedy_count = 0

    for _ in range(n_trials):
        action = epsilon_greedy_action(Q_values, epsilon, rng)

        if action == greedy_action:
            greedy_count += 1

    frequency = greedy_count / n_trials
    greedy_frequencies.append(frequency)

    print(
        f"Epsilon = {epsilon:.2f} | "
        f"Greedy frequency = {frequency:.4f}"
    )

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(epsilons, greedy_frequencies, marker="o")
plt.xlabel("Epsilon")
plt.ylabel("Greedy Action Frequency")
plt.title("Effect of Epsilon on Greedy Action Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(figures_dir, "epsilon_comparison.png"),
    dpi=300
)
plt.show()