import numpy as np
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])
def sample_next_state(current_state, P, rng):
    probabilities = P[current_state]
    next_state = rng.choice(
        len(probabilities),
        p=probabilities
    )
    return next_state
rng=np.random.default_rng(seed=42)
current_state= 0
states=[current_state]
for _ in range(30):
    current_state=sample_next_state(current_state, P, rng)
    states.append(current_state)
print("State sequence (Trình tự trạng thái):", states)