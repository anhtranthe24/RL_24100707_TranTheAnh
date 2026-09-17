import gymnasium as gym
env = gym.make("Blackjack-v1")
observation, info = env.reset()
print("Observation (Quan sát):", observation)
print("Info (Thông tin): ", info)
print("Observation space (Không gian quan sát):", env.observation_space)
print("Action space (Không gian hành động):", env.action_space)
env.close()
