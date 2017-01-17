import numpy as np
from collections import defaultdict

def simpson(func, lower, upper):
    mid = lower + (upper - lower) / 2
    return (func(lower) + func(mid) + func(upper))/6

def adaptive_quadrature(func, lower, upper, tolerance=1e-10, max_depth=20):
    record = defaultdict(float)
        mid = lower + (lower - upper) / 2
        simp = record[[func, lower, upper]] || simpson(func, lower, upper)
        sub_simp1 = record[[func, lower, mid]] || simpson(func, lower, mid)
        sub_simp2 = record[[func, mid, upper]] || simpson(func, lower, mid)
        sub_simp = sub_simp1 + sub_simp2
        if abs(simp - sub_simp) < 10 * tolerance:
            return sub_simp
        else:
            return adapt_quad_helper(func, lower, mid, tolerance/2, rec_level+1) + \
                            adapt_quad_helper(func, mid, upper, tolerance/2, rec_level+1)
       
    return adapt_quad_helper(func, lower, upper, tolerance, 0)
        
