import numpy as np
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])
print("Ma trận chuyển tiếp P (Transition Matrix P):")
print(P)
print("\nTổng của mỗi hàng (Sum of each row):")
print(P.sum(axis=1))