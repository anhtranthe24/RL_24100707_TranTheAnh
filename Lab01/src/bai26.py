import gymnasium as gym
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    env.reset()
    actions = [2, 2, 1, 1, 1, 2]
    for a in actions:
        env.step(a)
        print(env.render())
    env.close()