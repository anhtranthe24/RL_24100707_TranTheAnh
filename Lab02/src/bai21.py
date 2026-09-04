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
env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=True)
V = np.zeros(env.observation_space.n)
state = 0
action = 2
gamma = 0.99
q = q_from_v(env,V,state,action,gamma)
print("State:", state)
print("Action:", action)
print("Q(s,a):", q)
env.close()