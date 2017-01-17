import numpy as np
from collections import defaultdict

def solve(A, b, *args, **kwargs):
    safe_kwargs = defaultdict(lambda: None)
    for key, value in kwargs.items():
        safe_kwargs[key] = value


    methods = {
            "lu" : lu,
            "jacobi": jacobi,
            "gauss seidel": gauss_seidel,
            "sor": successive_overrelaxation
            }

    configs = {
            "tolerance": 1e-10,
            "max_iter": 100,
            "method": "gauss seidel"
            }

    for key, value in configs.items():
        print(key, value)# safe_kwargs[key])
        configs[key] = safe_kwargs[key] or value

    return methods[configs["method"]](A, b, *args, **configs)

def lu(A, b):
    m, n = np.shape(A)
    assert n == m, "Input matrix has dimension {}x{} and isn't square.".format(m,n)

    perm = np.eye(n)
    lower = np.zeros(n)
    upper = A.copy()

    for j in range(n):
        index = np.argmax(np.abs(upper[j:n, j]))
        assert upper[index, j] != 0.0, "This matrix is singular!"
        index += j

        if index > j:
            upper[[index, j]] = upper[[j, index]]
            perm[[index, j]] = perm[[j, index]]
            lower[[index, j]] = lower[[j, index]]

        k = slice(j+1, n)
        js = slice(j, n)
        lower[k, j] = upper[k, j] / upper[j, j]
        upper[k, js] -= lower[k,j] * upper[j, js]
    lower += np.eye(n)

    return perm, lower, upper
        


def _iteration_submethod(A, b, x, i):
    return 1./A[i,i] * (b[i] + A[i,i] * x[i] - A[i,:] @ x)

def _generic_iteration(A, b, subfunc, *args, tolerance=1e-10, max_iter=100):
    x = np.zeros(A.shape[0])
    iterations = 0

    while iterations < max_iter:
        for i in range(x.size):
            x[i] = subfunc(A, b, x, i, *args)
        err = np.linalg.norm(A @ x - b)
        if err < tolerance:
            break
        iterations += 1

    return x, iterations




def jacobi(A, b, *args, tolerance=1e-10, max_iter=100):
    x = np.zeros(A.shape[0])
    iterations = 0

    while iterations < max_iter:
        xprev = x.copy()
        for i in range(x.size):
            x[i] = _iteration_submethod(A, b, xprev, i)
        err = np.linalg.norm(A @ x - b)
        if err < tolerance:
            break
        iterations += 1

    return x, iterations

def gauss_seidel(A, b, *args, tolerance=1e-10, max_iter=100):
    def gauss_seidel_subfunc(A, b, x, i, *args):
        return _iteration_submethod(A, b, x, i)

    return _generic_iteration(A,b, gauss_seidel_subfunc, 
            tolerance=tolerance, max_iter=max_iter)
            
def successive_overrelaxation(A, b, omega = 1.01, *args, tolerance=1e-10, max_iter=100):
    def sor_subfunc(A, b, x, i, omega):
        return omega * _iteration_submethod(A, b, x, i) + (1 - omega) * x[i]

    return _generic_iteration(A, b, sor_subfunc, omega, tolerance=1e-10, max_iter=100)
