import gymnasium as gym 
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
print(f"Observation: {observation}")
print(f"Type: {type(observation)}")
print(f"Shape: {observation.shape}")
print(f"Info: {info}")
env.close()