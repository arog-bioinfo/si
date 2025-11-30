import numpy as np

def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    It computes the Tanimoto similarity of a point (x) to a set of points y.
        similarity_x_y1 = (x • y1) / (||x||^2 + ||y1||^2 - x • y1)
        similarity_x_y2 = (x • y2) / (||x||^2 + ||y2||^2 - x • y2)
        ...

    For zero vectors, we follow the convention that similarity is 1 when both vectors
    are zero (identical), and 0 when only one vector is zero (completely dissimilar).

    Parameters
    ----------
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.
    Returns
    -------
    np.ndarray
        Tanimoto similarity for each point in y.
    """
    # Compute dot products between x and each point in y
    dot_products = np.dot(y, x)

    # Compute the sum of squares for x and each point in y
    sum_squares_x = np.sum(x**2)
    sum_squares_y = np.sum(y**2, axis=1)

    # Compute the denominator: ||x||^2 + ||y||^2 - x • y
    denominator = sum_squares_x + sum_squares_y - dot_products

    # Handle division by zero cases (extra)
    # When both x and y_i are zero vectors, similarity is 1
    # When only one is a zero vector, similarity is 0
    both_zero = (sum_squares_x == 0) & (sum_squares_y == 0)
    x_zero = (sum_squares_x == 0) & (sum_squares_y != 0)
    y_zero = (sum_squares_x != 0) & (sum_squares_y == 0)

    # Initialize similarity array
    similarity = np.zeros_like(dot_products, dtype=float)

    # Compute similarity for non-zero denominator cases
    non_zero_mask = denominator != 0
    similarity[non_zero_mask] = dot_products[non_zero_mask] / denominator[non_zero_mask]

    # Set special cases
    similarity[both_zero] = 1.0  # Both vectors are zero
    similarity[x_zero | y_zero] = 0.0  # Only one vector is zero
    
    return similarity
