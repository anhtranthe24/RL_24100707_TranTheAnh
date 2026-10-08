import gymnasium as gym
import numpy as np
from bai25 import mc_action_value_prediction, stick_on_20_policy

def greedy_action(Q, state, n_actions):
    q_values = np.full(n_actions, -np.inf)

    if state in Q:
        for action, value in Q[state].items():
            q_values[action] = value
    else:
        return 0

    return int(np.argmax(q_values))

def main():
    env = gym.make("Blackjack-v1")
    Q = mc_action_value_prediction(env, stick_on_20_policy, 10000)

    test_states = [
        (20, 10, False),
        (18, 6, False),
        (13, 2, False),
        (18, 6, True)
    ]

    for state in test_states:
        action = greedy_action(Q, state, env.action_space.n)
        action_name = "Stick" if action == 0 else "Hit"
        print(f"State: {state} | Greedy Action: {action} ({action_name})")

    env.close()

if __name__ == "__main__":
    main()