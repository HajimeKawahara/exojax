def is_sorted(x):
    """Check if a list is sorted in ascending or descending order.

    Args:
        x: List to check.

    Returns:
        'ascending' if the list is sorted in ascending order,
        'descending' if the list is sorted in descending order,
        'unordered' otherwise.
    """
    if all(a <= b for a, b in zip(x, x[1:])):
        return 'ascending'
    elif all(a >= b for a, b in zip(x, x[1:])):
        return 'descending'
    else:
        return 'unordered'
