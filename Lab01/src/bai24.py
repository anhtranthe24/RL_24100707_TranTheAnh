import gymnasium as gym
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    env.reset()
    print(env.render())
    env.close()