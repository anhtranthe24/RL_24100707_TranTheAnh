import gymnasium as gym
def run_one_step(env, action):
    return env.step(action)
env = gym.make("CartPole-v1")
env.reset(seed=42)
print("Test Run_One_step Voi 5 Action")
for i in range(1, 6):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = run_one_step(env, action)
    print(f"Step: {i:2d} | Action: {action} | Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}")
env.close()