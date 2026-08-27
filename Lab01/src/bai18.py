import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
def run_random_agent(env, max_steps=500):
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward
def run_heuristic_agent(env, max_steps=500):
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = 1 if obs[2] > 0 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    num_episodes = 100  
    print(f"Đang thu thập dữ liệu từ {num_episodes} episodes cho 2 Agents")
    random_rewards = [run_random_agent(env) for _ in range(num_episodes)]
    heuristic_rewards = [run_heuristic_agent(env) for _ in range(num_episodes)]
    env.close()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes + 1), random_rewards, label='Random Agent (Ngẫu nhiên)', color='teal', alpha=0.6)
    plt.plot(range(1, num_episodes + 1), heuristic_rewards, label='Heuristic Agent (Logic cơ sở)', color='orange', alpha=0.8)
    plt.axhline(y=np.mean(random_rewards), color='teal', linestyle='--', linewidth=2, label=f'Mean Random: {np.mean(random_rewards):.2f}')
    plt.axhline(y=np.mean(heuristic_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean Heuristic: {np.mean(heuristic_rewards):.2f}')
    plt.title("Báo Cáo Tổng Hợp: So Sánh Hiệu Suất Random vs Heuristic", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Tổng điểm (Total Reward)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    print("Đang kết xuất biểu đồ tổng hợp. Kiểm tra!")
    plt.show()