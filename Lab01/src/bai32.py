import gymnasium as gym
import numpy as np
def improved_policy(observation):
    pole_angle = observation[2]
    pole_velocity = observation[3]
    return 1 if pole_angle + pole_velocity > 0 else 0
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(100):
        obs, _ = env.reset()
        tot = 0
        while True:
            obs, r, term, trunc, _ = env.step(improved_policy(obs))
            tot += r
            if term or trunc: break
        rewards.append(tot)
    print(f"Mean Reward Improved Policy: {np.mean(rewards)}")
    env.close()