import gymnasium as gym
def random_agent_v2(env, max_steps=500):
    env.reset()
    total_reward = 0.0
    episode_length = 0   
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        if terminated or truncated:
            if terminated:
                print("Lý do kết thúc: Termination (Cột ngã hoặc xe lệch khỏi ray)")
            elif truncated:
                print("Lý do kết thúc: Truncation (Đạt giới hạn thời gian/số bước)")
            break
    return total_reward, episode_length
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    reward, length = random_agent_v2(env)
    print(f"Episode length: {length}")
    print(f"Total reward: {reward}")
    env.close()