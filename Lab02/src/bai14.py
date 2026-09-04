import numpy as np
policy = np.array([1,0])
def print_policy(policy):
    for state, action in enumerate(policy):
        print(f"State {state} -> Action {action}")
print_policy(policy)