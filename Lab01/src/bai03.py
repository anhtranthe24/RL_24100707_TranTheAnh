import gymnasium as gym 
env = gym.make("CartPole-v1")
print(env.action_space)
print("Number of actions:", env.action_space.n)