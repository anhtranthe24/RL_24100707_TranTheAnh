import numpy as np
n_states = 2
n_actions = 2
policy = np.ones((n_states, n_actions)) / n_actions
print("Policy:", policy)
print("\nSum of action probabilities:")
for state in range(n_states):
    print(f"State {state}: ",f"{policy[state].sum()}")
valid = np.allclose(policy.sum(axis=1),1.0)
print("\nPolicy valid:", valid)