import gymnasium as gym
def random_agent(env, max_steps=500):
    env.reset()
    total_reward = 0.0
    episode_length = 0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        if terminated or truncated:
            break
    return total_reward, episode_length
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    reward, length = random_agent(env)
    print("--- Kết quả của Random Agent ---")
    print(f"Episode length: {length}")
    print(f"Total reward: {reward}")
    env.close()