import gymnasium as gym
import matplotlib.pyplot as plt
from mdp_utils import value_iteration
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
V, n_iterations, deltas = value_iteration(env)
print("Giá trị tối ưu:", V)
print("Số vòng lặp:", n_iterations)
print("Delta cuối:", deltas[-1])
plt.plot(deltas, label="Delta")
plt.title("Value Iteration Convergence")
plt.xlabel("Iteration")
plt.ylabel("Delta")
plt.grid(True)
plt.legend()
plt.savefig("./figures/value_iteration_convergence.png", dpi=300, bbox_inches="tight")
plt.show()
env.close()