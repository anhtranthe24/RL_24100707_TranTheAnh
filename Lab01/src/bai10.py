import gymnasium as gym
env = gym.make("CartPole-v1")
env.reset(seed=42)
total_reward = 0.0
episode_length = 0
for t in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_length += 1
    if terminated or truncated:
        break
print(f"Episode length: {episode_length}")
print(f"Total reward: {total_reward}")
env.close()