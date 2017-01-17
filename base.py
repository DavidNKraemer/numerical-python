import numpy as np

class NumericalAlgorithm:
    pass

class NumericalIterator(NumericalAlgorithm):
    pass

class FPIterator(NumericalIterator):
    TOLERANCE = 0.5e-8
    MAX_ITER = 100


    def __init__(self, update_func, evaluate_func):
        self.updator = update_func
        self.evaluator = evaluate_func

    def __call__(self, func, guess, tolerance=TOLERANCE, max_iter=MAX_ITER):
        guess = np.asarray(guess)
        fixed_points = np.empty((max_iter + 1, *guess.shape))
        fixed_points[0] = guess
        iterations = 1

        while iterations < max_iter:
            fixed_points[iterations] = self.updator(func, fixed_points[iterations - 1])
            if self.evaluator(fixed_points[iterations], fixed_points[iterations-1], tolerance):
                break
            iterations += 1

        fixed_points = fixed_points[:iterations + 1]
        errors = np.abs(fixed_points[1:] - fixed_points[:-1])

        return fixed_points, errors, iterations

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "FPIterator."

def _fp_update(func):
    """
    x[i] = f(x[i-1])
    """
    return lambda x: func(x)

def _fp_evaluate(newer, older, tolerance):
    """
    |x[i] - x[i-1]| < tol
    """
    return np.abs(newer - older) < tolerance

fixed_point = FPIterator(_fp_update, _fp_evaluate)

def _newton_update(func, fderiv):
    """
    x[i] = x - f(x[i-1]) / f'(x[i-1])
    """
    return lambda x: x - func(x) / fderiv(x)

newton = FPIterator(_newton_update, _fp_evaluate)
