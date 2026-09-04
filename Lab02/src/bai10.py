import numpy as np
import matplotlib.pyplot as plt
def compute_return(rewards, gamma):
    G=0.0
    for t, reward in enumerate(rewards):
        G+=(gamma**t)*reward
    return G
rewards = [0,0,0,0,10]
gammas=np.linspace(0,1,101)
returns = []
for gamma in gammas:
    G= compute_return(rewards, gamma)
    returns.append(G)
plt.figure(figsize=(8,5))
plt.plot(gammas, returns, label="GO")
plt.title("Effect of Discount Factor Gamma (Ảnh hưởng của hệ số chiết khấu Gamma)")
plt.xlabel("Gamma (Hệ số chiết khấu)")
plt.ylabel("GO")
plt.grid(True)
plt.savefig("./figures/gamma_comparison.png", dpi=300, bbox_inches="tight")
plt.show()