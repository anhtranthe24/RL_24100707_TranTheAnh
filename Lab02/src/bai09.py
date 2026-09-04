def discounted_return(rewards, gamma):
    G = 0.0
    returns = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns
rewards = [0,0,0,1]
gamma=0.9
returns=discounted_return(rewards, gamma)
print("Rewards (Phần thưởng):\n", rewards)
print("Gamma:\n", gamma)
print("Returns: \n", returns)