import numpy as np
A = np.array([
[1,2,3],
[2,3,4],
[4,5,6],
[1,1,1]])

print("A =\n", A, "\n")
u, s, vt = np.linalg.svd(A)
print("u =\n", u)
print("s =\n", s)
print("vt =\n", vt)