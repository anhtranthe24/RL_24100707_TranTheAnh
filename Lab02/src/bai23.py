import gymnasium as gym
import numpy as np
def q_from_v(env, V, state, action, gamma):
    P = env.unwrapped.P
    q_value = 0.0
    for probability, next_state, reward, terminated in P[state][action]:
        if terminated:
            future_value = 0.0
        else:
            future_value = V[next_state]
        q_value += probability * (reward + gamma * future_value)
    return q_value
def policy_evaluation_sweep(env,policy,V,gamma):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    new_V = np.zeros(n_states)
    for state in range(n_states):
        value = 0.0
        for action in range(n_actions):
            q_value = q_from_v(env,V,state,action,gamma)
            value += (policy[state][action]* q_value)
        new_V[state] = value
    return new_V
env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n
policy = np.ones(
    (n_states, n_actions)
) / n_actions
V = np.zeros(n_states)
gamma = 0.99
new_V = policy_evaluation_sweep(env,policy,V,gamma)
print("Old V:")
print(V)
print("\nNew V after one sweep:")
print(new_V)
env.close()