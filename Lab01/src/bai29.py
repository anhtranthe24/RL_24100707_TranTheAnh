import gymnasium as gym
def policy(observation, env):
    return env.action_space.sample()
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    action = policy(obs, env)
    print("Action từ policy:", action)
    env.close()