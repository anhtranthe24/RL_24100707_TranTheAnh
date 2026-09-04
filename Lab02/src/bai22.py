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
def action_values(env, V, state, gamma):
    n_actions = env.action_space.n
    q_values = np.zeros(n_actions)
    for action in range(n_actions):
        q_values[action] = q_from_v(env,V,state,action,gamma)
    return q_values
env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=True)
V = np.zeros(env.observation_space.n)
state = 0
gamma = 0.99
q_values = action_values(env,V,state,gamma)
print("State:", state)
print("Q-values:", q_values)
env.close()