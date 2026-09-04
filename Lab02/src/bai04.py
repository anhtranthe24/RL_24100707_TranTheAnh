import numpy as np
import matplotlib.pyplot as plt
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])
def state_distribution(p0, P, n_steps):
    p = p0.copy()
    for _ in range(n_steps):
        p = p @ P
    return p
p0 = np.array([1.0, 0.0, 0.0])
steps = [1, 2, 5, 10, 50]
distributions = []
for n in steps:
    p = state_distribution(p0, P, n)
    distributions.append(p)
    print(f"t = {n}", p)
    print("Sum =", p.sum())
    print()
distributions = np.array(distributions)
plt.figure(figsize=(8, 5))
plt.plot(steps, distributions[:, 0], marker="o", label="Sunny")
plt.plot(steps, distributions[:, 1], marker="o", label="Cloudy")
plt.plot(steps, distributions[:, 2], marker="o", label="Rainy")
plt.title("Markov State Distribution")
plt.xlabel("Number of Steps")
plt.ylabel("Probability")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./figures/markov_distribution.png", dpi=300, bbox_inches="tight")
plt.close()