import gymnasium as gym
import numpy as np
env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=True)
P = env.unwrapped.P
all_valid = True
for state in range(env.observation_space.n):
    for action in range(env.action_space.n):
        probabilities = [transition[0]for transition in P[state][action]]
        total = sum(probabilities)
        valid = np.isclose(total,1.0)
        if not valid:
            print(
                f"Invalid transition at "
                f"state={state}, "
                f"action={action}"
            )
            all_valid = False
print("All transitions valid:", all_valid)
env.close()