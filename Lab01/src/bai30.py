import gymnasium as gym
import numpy as np
def always_left_policy(observation): return 0
def always_right_policy(observation): return 1
def run_policy(env, policy_func):
    obs, _ = env.reset()
    tot = 0
    while True:
        obs, r, term, trunc, _ = env.step(policy_func(obs))
        tot += r
        if term or trunc: break
    return tot
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    left_rewards = [run_policy(env, always_left_policy) for _ in range(100)]
    right_rewards = [run_policy(env, always_right_policy) for _ in range(100)]
    print(f"Mean Left: {np.mean(left_rewards)}, Mean Right: {np.mean(right_rewards)}")
    env.close()