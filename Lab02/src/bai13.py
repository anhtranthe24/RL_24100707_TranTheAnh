import numpy as np
def validate_mdp(P, n_states, n_actions):
    for state in range(n_states):
        for action in range(n_actions):
            transitions = P[state][action]
            total = sum(t[0] for t in transitions)
            if not np.isclose(total, 1.0):
                print(f"Invalid transition at state={state}, action={action}")
                return False
    return True
n_states = 2
n_actions = 2
P = [
    [[(1.0, 1, 1, False)], [(1.0, 0, 0, False)]],
    [[(1.0, 0, 2, False)], [(1.0, 1, 3, True)]]
]
print("MDP valid:", validate_mdp(P, n_states, n_actions))