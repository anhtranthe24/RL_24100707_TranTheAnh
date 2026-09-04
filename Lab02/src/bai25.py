import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
def q_from_v(env, V, state, action, gamma):
    q = 0.0
    for p, next_state, reward, terminated in env.unwrapped.P[state][action]:
        q += p * (reward + gamma * (0 if terminated else V[next_state]))
    return q
def policy_evaluation(env, policy, gamma=0.99, theta=1e-8, max_iterations=10000):
    V = np.zeros(env.observation_space.n)
    deltas = []
    for _ in range(max_iterations):
        new_V = np.zeros_like(V)
        for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                new_V[state] += policy[state][action] * q_from_v(env, V, state, action, gamma)
        delta = np.max(np.abs(new_V - V))
        deltas.append(delta)
        V = new_V
        if delta < theta:
            break
    return V, deltas
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
V, deltas = policy_evaluation(env, policy)
plt.plot(deltas, label="Delta")
plt.title("Policy Evaluation Convergence")
plt.xlabel("Iteration")
plt.ylabel("Delta")
plt.grid(True)
plt.legend()
plt.savefig("./figures/policy_iteration_convergence.png", dpi=300, bbox_inches="tight")
plt.show()
env.close()