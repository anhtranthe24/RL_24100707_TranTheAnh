import gymnasium as gym
import numpy as np
def evaluate_seed(target_seed, num_episodes=20):
    env = gym.make("CartPole-v1")
    rewards = []   
    for _ in range(num_episodes):
        env.reset(seed=target_seed)
        total_reward = 0.0       
        for _ in range(500):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward          
            if terminated or truncated:
                break                
        rewards.append(total_reward)      
    env.close()
    return np.mean(rewards)
if __name__ == "__main__":
    mean_seed_42 = evaluate_seed(42)
    mean_seed_100 = evaluate_seed(100)
    print(f"Đánh giá độ lệch Reward theo Seed")
    print(f"Reward trung bình với seed=42 : {mean_seed_42:.2f}")
    print(f"Reward trung bình với seed=100: {mean_seed_100:.2f}")