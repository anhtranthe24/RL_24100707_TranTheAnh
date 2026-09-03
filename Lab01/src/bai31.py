import gymnasium as gym
import numpy as np
def angle_based_policy(observation):
    return 1 if observation[2] > 0 else 0
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(100):
        obs, _ = env.reset()
        tot = 0
        while True:
            obs, r, term, trunc, _ = env.step(angle_based_policy(obs))
            tot += r
            if term or trunc: break
        rewards.append(tot)
    print(f"Mean Reward Angle Policy: {np.mean(rewards)}")
    env.close()