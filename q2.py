import numpy as np
from math import sqrt
A = np.empty((1401,1401))
x = 0
y = 0
delta = 0.001
count = 0

for i in range(0,len(A)):
	x = -0.7 + delta * (i-1)
	for j in range (0,len(A[0])):
		y = -0.7 + delta * (j-1)
		A[i][j] = (sqrt(1 - (x**2) - (y**2)))
		
u, s, vt = np.linalg.svd(A)

A2 = np.empty((len(u), len(vt)))
rank = 2
for i in range(rank):
	A2 += s[i] * np.outer(u.T[i], vt[i])
 
print("||A - A2|| = ", np.linalg.norm(A - A2))