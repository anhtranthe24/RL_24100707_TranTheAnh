import numpy as np
def compute_return(rewards, gamma):
    G=0.0
    for t, reward in enumerate(rewards):
        G+=(gamma**t)*reward
    return G
sequence_A = [5, 0, 0, 0, 0]
sequence_B = [0, 0, 0, 0, 10]
gammas = np.linspace(0,1,10001)
better_gammas=[]
for gamma in gammas:
    return_A = compute_return(sequence_A, gamma)
    return_B = compute_return(sequence_B, gamma)
    if return_B > return_A:
        better_gammas.append(gamma)
print("Gamma range where B > A (Phạm vi gamma mà B > A):\n")
print(f"{better_gammas[0]:.6f}"
      f" < gamma < "
      f"{better_gammas[-1]:.6f}")