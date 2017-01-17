import numpy as np
from collections import defaultdict
from functools import reduce, partial
from operator import mul
from bisect import bisect_left

def _prod(iterable, accumulator=1):
    return reduce(mul, iterable, accumulator)

def cheby(i, n):
    return np.cos((2 * i + 1) * np.pi / (2 * n))

_cheby_points = defaultdict(partial(np.ndarray, 0))
def chebyshev_points(degree):
    assert(degree >= 0)

    if _cheby_points[degree].size == 0:
        _cheby_points[degree] = np.sort(cheby(np.arange(degree),degree))

    return _cheby_points[degree]

def newton(x, y, dtype=float):
    assert(x.shape == y.shape)
    n = x.size
    
    table = np.zeros((n, n), dtype=dtype)
    table[:,0] = y.copy()

    for j in range(1,n):
        for i in range(n-j):
            table[i,j] = dtype((table[i+1,j-1] - table[i,j-1]) / (x[i+j] - x[i]))

    return table[0]

def lagrange(x, y):
    assert(x.shape == y.shape)
    n = x.size

    def p(val):
        summand = 0.0
        for k in range(n):
            product = 1.0
            for j in range(n):
                if j != k:
                    product *= ((val - x[j]) / (x[k] - x[j]))
            summand += y[k] * product
        return summand 

    return p

def nest(coeffs, values):
    def p(x):
        value = coeffs[0]
        product = 1.0
        for (i, c) in enumerate(coeffs[1:]):
            product *= (x - values[i])
            value += c * product

        return value
    return p

def _cubic_spline(x, y, end_condition):
    assert(x.shape == y.shape)
    n = x.size

    A = np.zeros((n,n))

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]

    r = np.zeros(n)
    r[1:-1] = 3 * ( dy[1:]/dx[1:] - dy[:-1]/dx[:-1])

    A.flat[n+1:-n:n+1] = 2 * (dx[:-1] + dx[1:])
    A.flat[n:-n:n+1] = dx[:-1]
    A.flat[n+2:-n:n+1] = dx[1:]

    end_condition(A, r)


    coeffs = np.zeros((n,3))
    coeffs[:,1] = np.linalg.solve(A,r)
    coeffs[:-1,0] = dy / dx - dx * (2*coeffs[:-1,1] + coeffs[1:,1]) / 3
    coeffs[:-1,2] = (coeffs[1:,1] - coeffs[:-1,1]) / (3 * dx)

    return _spline_func(x, y, coeffs[:-1,:])

def _spline_func(points, samples, coeffs):
    print(coeffs)
    def p(x):
        x = np.asarray(x)
        vals = np.empty(x.shape)

        too_low = x < points[0]
        too_high = x >= points[-1]
        in_bounds = (x >= points[0]) & (x <= points[-1])

        print(in_bounds)
        
        low_diff = x[too_low] - points[0]
        high_diff = x[too_high] - points[-1]

        basis = lambda x: np.array([x, x*x, x*x*x])

        vals[too_low] = coeffs[0].dot(basis(low_diff)) + samples[0]
        vals[too_high] = coeffs[-1].dot(basis(high_diff)) + samples[-1]

        in_diff = x[in_bounds]- points[i]

        vals[in_bounds] = coeffs[i].dot(basis(in_diff)) + samples[i]

        return vals

    return p

def _natural_endpt_cond(A, r):
    A.flat[[0, -1]] = 1

def _curv_adj_endpt_cond(A, r, v1, vn):
    A.flat[[0, -1]] = 2
    r[0] = v1
    r[-1] = vn

def _clamped_adj_endpt_cond(A, r, v1, vn, dx1, dxn, dy1, dyn):
    A.flat[0] = 2 * dx1
    A.flat[1] = dx1
    A.flat[-1] = 2 * dxn
    A.flat[-2] = dxn

    r[0] = 3 * (dy1 / dx1 - v1)
    r[-1] = 3 * (vn - dyn / dxn)

def _parabolic_endpt_cond(A, r):
    A.flat[[0, -2]] = 1
    A.flat[[1, -1]] = -1

def _not_knot_endpt_cond(A, r, dx1, dx2, dxn1, dxn2):
    A.flat[[0, -1]] = dx2, dxn2
    A.flat[[1, -2]] = -(dx1 + dx2), -(dxn1 + dxn2)
    A.flat[[2, -3]] = dx1, dxn1
    

def natural_spline(x, y):
    return _cubic_spline(x, y, lambda A,r: _natural_endpt_cond(A, r))

def curvature_adjusted_spline(x, y, v1, vn):
    return _cubic_spline(x, y, lambda A,r: _curv_adj_endpt_cond(A, r, v1, vn))

def clamped_spline(x, y, v1, vn):
    (dx1, dxn) = (x[1] - x[0], x[-1] - x[-2])
    (dy1, dyn) = (y[1] - y[0], y[-1] - y[-2])

    return _cubic_spline(x, y, 
            lambda A,r: _clamped_adj_endpt_cond(A, r, v1, vn, dx1, dxn, dy1, dyn))

def parabolic_spline(x, y):
    return _cubic_spline(x, y, parabolic_endpt_cond)

def not_knot_spline(x, y):
    (dx1, dx2) = (x[1] - x[0], x[2] - x[1])
    (dxn1, dxn2) = (x[-1] - x[-2], x[-2] - x[-3])

    return _cubic_spline(x, y, 
            lambda A, r: _not_knot_endpt_cond(dx1, dx2, dxn1, dxn2))




