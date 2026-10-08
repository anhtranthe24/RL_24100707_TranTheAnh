import numpy as np

def update_incremental_mean(Q, N, state, action, G):
    N[state][action] += 1
    Q[state][action] += (
        G - Q[state][action]
    ) / N[state][action]

Q = {
    (18, 6, False): np.array([0.0, 0.0])
}

N = {
    (18, 6, False): np.array([0, 0])
}

state = (18, 6, False)
action = 1

returns = [1.0, -1.0, 1.0]

for G in returns:
    update_incremental_mean(Q, N, state, action, G)
    print(
        f"G: {G:.1f} | "
        f"N: {N[state][action]} | "
        f"Q: {Q[state][action]:.4f}"
    )