import gymnasium as gym
import numpy as np
def experiment(seed, n_episodes):
    env = gym.make("CartPole-v1")
    rewards = []   
    for _ in range(n_episodes):
        env.reset(seed=seed)
        total_reward = 0.0     
        for _ in range(500):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break                
        rewards.append(total_reward)     
    env.close() 
    return {
        "seed": seed,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards)
    }
if __name__ == "__main__":
    seeds = [42, 100, 777, 2026, 9999]
    num_episodes = 20
    for s in seeds:
        result = experiment(s, num_episodes)
        print(result)