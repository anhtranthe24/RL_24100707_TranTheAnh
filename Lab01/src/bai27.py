import gymnasium as gym
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    success = 0
    total_episodes = 100
    for _ in range(total_episodes):
        env.reset()
        while True:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                if r == 1.0: success += 1
                break
    print(f"Success rate: {success / total_episodes}")
    env.close()