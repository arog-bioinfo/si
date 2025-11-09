import numpy as np

def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    It computes the Tanimoto similarity of a point (x) to a set of points y.
        similarity_x_y1 = (x • y1) / (||x||^2 + ||y1||^2 - x • y1)
        similarity_x_y2 = (x • y2) / (||x||^2 + ||y2||^2 - x • y2)
        ...
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

    # Compute Tanimoto similarity
    similarity = dot_products / denominator

    return similarity
