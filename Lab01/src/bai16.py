import gymnasium as gym
import numpy as np
def heuristic_agent(obs):
    return 1 if obs[2] > 0 else 0
def run_heuristic(env, max_steps=500):
    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = heuristic_agent(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break    
    return total_reward
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    num_episodes = 100
    rewards = []
    print(f"Đang kiểm thử Heuristic qua {num_episodes} episodes")    
    for _ in range(num_episodes):
        rewards.append(run_heuristic(env))        
    rewards_array = np.array(rewards)   
    print("\nBáo cáo Thống kê Heuristic Agent")
    print(f"Điểm trung bình (Mean)  : {np.mean(rewards_array):.2f}")
    print(f"Độ lệch chuẩn (Std Dev) : {np.std(rewards_array):.2f}")
    print(f"Điểm cao nhất (Max)     : {np.max(rewards_array)}")
    print(f"Điểm thấp nhất (Min)    : {np.min(rewards_array)}")    
    env.close()