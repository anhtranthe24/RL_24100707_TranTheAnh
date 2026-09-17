from mc_utils import compute_returns
rewards = [0, 0, 1]
returns = compute_returns(rewards, gamma=1.0)
print("Reward (Phần thưởng):", rewards)
print("Returns (Trả về):", returns)
