import gymnasium as gym
env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n
observation, info = env.reset()
print("Number of states:", n_states)
print("Number of actions:", n_actions)
print("Initial observation:", observation)
env.close()