import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
def heuristic_agent(obs):
    return 1 if obs[2] > 0 else 0
def run_heuristic(env, max_steps=500):
    obs, _ = env.reset()
    total_reward = 0.0  
    for _ in range(max_steps):
        action = heuristic_agent(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward        
        if terminated or truncated:
            break           
    return total_reward
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    num_episodes = 100
    rewards = []
    print(f"Đang thu thập dữ liệu từ {num_episodes} episodes")
    for _ in range(num_episodes):
        rewards.append(run_heuristic(env))    
    env.close()
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), rewards, marker='o', linestyle='-', color='orange', alpha=0.7)
    plt.title("Hiệu suất của Heuristic Agent qua 100 Episodes", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Tổng điểm (Total Reward)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    mean_reward = np.mean(rewards)
    plt.axhline(y=mean_reward, color='r', linestyle='-', label=f'Mean: {mean_reward:.2f}')
    plt.legend()
    print("Đang kết xuất biểu đồ. Kiểm tra cửa sổ giao diện đồ họa.")
    plt.show()