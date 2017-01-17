import numpy as np
import linalg

n = 5
A = np.random.rand(n,n)
x = np.ones(n)
b = A @ x

print(linalg.solve(A,b))
