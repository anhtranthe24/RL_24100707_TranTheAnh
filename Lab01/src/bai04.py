import gymnasium as gym
env = gym.make("CartPole-v1")
obs_space = env.observation_space
print(f"Observation space: {obs_space}")
print(f"Shape: {obs_space.shape}")
print(f"Type: {obs_space.dtype}")
print(f"Low bounds:\n{obs_space.low}")
print(f"High bounds:\n{obs_space.high}")
