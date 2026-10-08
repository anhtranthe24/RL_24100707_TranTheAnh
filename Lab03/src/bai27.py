import numpy as np

def epsilon_greedy_action(Q, state, n_actions, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))

    q_values = np.full(n_actions, -np.inf)

    if state in Q:
        for action, value in Q[state].items():
            q_values[action] = value

    return int(np.argmax(q_values))

Q = {
    (0,): {
        0: 1.0,
        1: 5.0
    }
}

rng = np.random.default_rng(42)

for _ in range(10):
    action = epsilon_greedy_action(
        Q,
        (0,),
        n_actions=2,
        epsilon=0.1,
        rng=rng
    )
    print("Action:", action)