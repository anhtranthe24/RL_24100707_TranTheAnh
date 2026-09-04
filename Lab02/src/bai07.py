def compute_return(rewards, gamma):
    G=0.0
    for t, reward in enumerate(rewards):
        G+=(gamma**t)*reward
    return G
rewards=[1,1,1,1,1]
gamma=1.0
G=compute_return(rewards, gamma)
print("\nRewards (Phần thưởng):\n", rewards)
print("\nGamma:\n", gamma)
print("\nReturn (Giá trị hoàn trả):\n", G)