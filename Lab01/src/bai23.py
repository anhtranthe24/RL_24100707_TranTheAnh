import gymnasium as gym
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Số state:", env.observation_space.n)
    print("Số action:", env.action_space.n)
    env.close()