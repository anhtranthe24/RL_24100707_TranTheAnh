def compute_return(rewawrds, gamma):
    G=0.0
    for t, reward in enumerate(rewards):
        G+=(gamma**t)*reward
    return G
rewards=[1,1,1,1,1]
gammas=[0.0, 0.5, 0.9, 0.99,1.0]
print("Gamma | Return")
for gamma in gammas:
    G=compute_return(rewards, gamma)
    print(f"{gamma:.2f}  | {G:.6f}")