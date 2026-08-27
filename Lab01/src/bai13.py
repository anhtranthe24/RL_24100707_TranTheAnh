import gymnasium as gym
import numpy as np
def random_agent_v2(env, max_steps=500):
    env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    num_episodes = 100
    rewards = []
    print(f"--Đang triển khai {num_episodes} episodes --")
    for i in range(num_episodes):
        reward = random_agent_v2(env)
        rewards.append(reward)
    reward_array = np.array(rewards)
    mean_reward = np.mean(reward_array)
    std_reward = np.std(reward_array)
    max_reward = np.max(reward_array)
    min_reward = np.min(reward_array)
    print("\n Báo cáo thống kê thực nghiệm")
    print(f"Mean: {mean_reward:.2f}")
    print(f"Std Dev: {std_reward:.2f}")
    print(f"Max: {max_reward}")
    print(f"Min: {min_reward}")
    env.close()