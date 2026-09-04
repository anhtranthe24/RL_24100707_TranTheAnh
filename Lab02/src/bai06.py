import numpy as np
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])
def sample_next_state(current_state, P, rng):
    probabilities=P[current_state]
    return rng.choice(len(probabilities), p =probabilities )
def state_distribution(P, initial_state, steps):
    p =p0.copy()
    for _ in range(steps):
        p = p@P
    return p
rng=np.random.default_rng(seed=42)
n_transitions= 100000
current_state= 0
counts = np.zeros(3, dtype=int)
for _ in range(n_transitions):
    current_state=sample_next_state(current_state, P, rng)
    counts[current_state] += 1
simulated_distibution = counts/n_transitions
p0=np.array([1.0, 0.0, 0.0])
theoretical_distribution=state_distribution(P, p0, 1000)
print("\nSimulated distribution (Phân phối mô phỏng):\n", simulated_distibution)
print("\nTheoretical distribution (Phân phối lý thuyết):\n", theoretical_distribution)  
print("\nDifference (Sự khác biệt):\n", np.abs(simulated_distibution - theoretical_distribution))
