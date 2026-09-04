import numpy as np
def Xac_thuc_ma_tran_chuyen_tiep(P, tol=1e-10):
    if P.ndim != 2:
        return False
    rows, cols = P.shape
    if rows != cols:
        return False
    if not np.all((P>=0) & (P <= 1)):
        return False
    if not np.all(np.isclose(P.sum(axis=1), 1.0, atol=tol)):
        return False
    return True
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])
print("P:")
print(P)
print("\nCó hợp lệ không(Is valid) ?\n",Xac_thuc_ma_tran_chuyen_tiep(P))