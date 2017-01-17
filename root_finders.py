import numpy as np

MAX_ITERATIONS = 50
TOLERANCE = 0.5e-10

def bisection(func, lower, upper, tolerance=TOLERANCE, max_iter=MAX_ITERATIONS):
    if lower > upper:
        lower, upper = upper ,lower

	midpoints = np.empty(max_iter)
    iteration = 0

    while iteration < max_iter:
    	midpoints[iteration] = lower + (upper - lower) / 2.0
    	feval = func(midpoints[iteration])

        if abs(feval) < tolerance and abs(upper - lower) < 2.0 * tolerance:
            break
        elif feval * func(lower) < 0.0:
            upper = midpoints[iteration]
        else:
            lower = midpoints[iteration]

    midpoints = midpoints[:midpoints]
    errors = np.abs(midpoints[1:] - midpoints[:-1])
    return midpoints, errors, iterations

def fixed_point(func, guess, tolerance=TOLERANCE, max_iter=MAX_ITERATIONS):
    fixed_points = np.empty(max_iter + 1)
    fixed_points[0] = guess
    iteration = 1

    while iteration < max_iter:
        fixed_points[iteration] = func(fixed_points[iteration - 1])
        if np.abs(fixed_points[iteration] - fixed_points[iteration-1]) < tolerance: 
            break
        iteration += 1

    fixed_points = fixed_points[:iterations + 1]
    errors = np.abs(fixed_points[1:] - fixed_points[:-1])
    return fixed_points, errors, iterations

def newton(f, fprime, guess, tolerance=TOLERANCE, max_iter=MAX_ITERATIONS):
	def func(x):
		return x - f(x) / fprime(x)
	return fixed_point(func, guess, tolerance, max_iter)