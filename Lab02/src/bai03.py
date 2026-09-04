import numpy as np
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5] 
])
p0 = np.array([1.0, 0.0, 0.0])
p1 = p0 @ P
print("Initial distribution (Phân phối ban đầu):", p0)
print("\nDistribution after one step (Phân phối sau một bước):", p1)
print("\nSum(Tổng):", p1.sum())