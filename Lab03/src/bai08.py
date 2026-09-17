from mc_utils import compute_returns
rewards = [0, 0, 1]
gammas=[1.0, 0.9, 0.5]
for gamma in gammas:
    returns = compute_returns(rewards, gamma)
    print(f"Gamma = {gamma}: {returns}")