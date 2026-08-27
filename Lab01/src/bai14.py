import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
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
    reward = []
    print(f"Đang thu thập dữ liệu từ {num_episodes}")
    for _ in range(num_episodes):
        reward.append(random_agent_v2(env))
    env.close()
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), reward, marker='o', linestyle='-', color='teal', alpha=0.7)
    plt.title("Hiệu suất của Random Agent qua 100 Episodes", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Tổng điểm (Total Reward)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    mean_rewward = np.mean(reward)
    plt.axhline(y=mean_rewward, color='r', linestyle='-', label=f'Mean: {mean_rewward:.2f}')
    plt.legend()
    print("Đang kết xuất biểu đồ. Kiểm tra cửa sổ giao diện đồ hoạ.")
    plt.show()