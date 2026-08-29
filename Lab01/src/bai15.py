import gymnasium as gym
def heuristic_agent(obs):
    pole_angle=obs[2]
    return 1 if pole_angle > 0 else 0
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)
    total_reward=0.0
    episode_length=0
    for _ in range(500):
        action = heuristic_agent(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length+=1
        if terminated or truncated:
            break
    print("Kết quả của Heuristic Agent")
    print(f"Thời gian sinh tồn: {episode_length} bước")
    print(f"Tổng điểm đạt được: {total_reward}")
    env.close()