import gymnasium as gym
state = 0
action = 2 
for slippery in [False, True]:
    env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=slippery)
    P = env.unwrapped.P
    transitions = P[state][action]
    print("is_slippery =", slippery)
    print("Number of transitions:",
          len(transitions))
    for probability, next_state, reward, terminated in transitions:
        print(
            f"Probability = {probability:.4f}, "
            f"Next state = {next_state}, "
            f"Reward = {reward}, "
            f"Terminated = {terminated}"
        )
    env.close()