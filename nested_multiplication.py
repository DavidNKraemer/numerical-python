def nest(coefficients, x, base_points):
    """Evaluates polynomial from nested form using Horner's Method
    
    Parameters
    ----------
    coefficients, list
        This is the coefficients of the polynomial, with coefficients[0]
        indicating the constant term and coefficients[-1] indicating the degree
        of the polynomial term.
    x, float
        The value to evaluate the polynomial on
    base_points, list
        An array of base points, if needed.

        len(base_points) + 1 == len(coefficients)
        Default: base_points == [0] * (len(coefficients) - 1)

    Returns
    -------
    value, float
        `value` is the function value using the coefficients and x
    """
    value = float(coefficients[-1])
    for i, coef in enumerate(coefficients[-2::-1]):
        value = coef + value * (x - base_points[i])

    return value
    

