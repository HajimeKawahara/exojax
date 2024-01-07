import numpy as np


def is_sorted(x):
    """Check if a list is sorted in ascending or descending order.

    Args:
        x: List to check.

    Returns:
        'single' if x is not list, but a single value
        'ascending' if the list is sorted in ascending order,
        'descending' if the list is sorted in descending order,
        'unordered' otherwise.
    """

    if isinstance(x, (int, float, str, bool)):
        return "single"
    elif all(a <= b for a, b in zip(x, x[1:])):
        return "ascending"
    elif all(a >= b for a, b in zip(x, x[1:])):
        return "descending"
    else:
        return "unordered"


def is_outside_range(xarr, xs, xe):
    """
    Check if all elements in the array are outside the specified range.

    Args:
        xarr (numpy.ndarray): An array of numerical values.
        xs (float): The start of the range (exclusive).
        xe (float): The end of the range (exclusive).

    Returns:
        bool: True if all elements in xarr are outside the range (xs, xe), False otherwise.

    Examples:
        >>> xarr = np.array([1.2, 1.4, 1.7, 1.3, 1.0])
        >>> xs = 0.7
        >>> xe = 0.8
        >>> result = is_outside_range(xarr, xs, xe) #-> True

        
    """
    return not np.any((xarr > xs) & (xarr < xe))
