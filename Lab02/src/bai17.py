import gymnasium as gym
env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=True)
P = env.unwrapped.P
ACTION_NAMES = {0: "LEFT",1: "DOWN",2: "RIGHT",3: "UP"}
state = 0
for action in range(env.action_space.n):
    print(f"\nAction {action} ", f"({ACTION_NAMES[action]})")
    for probability, next_state, reward, terminated in P[state][action]:
        print(f"Probability = {probability:.4f}, "f"Next state = {next_state}, "f"Reward = {reward}, "f"Terminated = {terminated}")
env.close()