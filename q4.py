import numpy as np
from sympy import Matrix
#I have not used numpy/scipy for the entirety of this file, as scipy's built-in nullspace function provides a null space using fractions that resolve to floating-point numbers the computer cannot accurately represent
#sympy's nullspace function gives a null space that is human-readable and without these errors, meaning that A * nullspace_of_A will actually resolve to a zero matrix
#it's back to numpy for the pseudoinverse, though

A = Matrix([
[3,2,-1,4],
[1,0,2,3],
[-2,-2,3,-1]
])



nspace = A.nullspace()
print ("Null Space Vector 1:\n", np.array(nspace[0]), "\nNull Space Vector 2:\n", np.array(nspace[1]))

a = np.array([
[3,2,-1,4],
[1,0,2,3],
[-2,-2,3,-1]])

print ("\nA is a 3x4 matrix; it has a nullity of 2. Per rank-nullity theorem, we know its rank is 4 - 2 = 2.\nThus, its columns are not linearly independent in R3, nor are its rows linearly independent in R4.\n\nInverse of A:\n",
np.linalg.pinv(a))