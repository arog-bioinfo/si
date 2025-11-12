import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the root mean squared error for the y_pred variable.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset.
    y_pred: np.ndarray
        The predicted labels of the dataset.

    Returns
    -------
    rmse: float
        The root mean squared error of the model.
    """
    mse = np.sum((y_true - y_pred) ** 2) / len(y_true)
    return np.sqrt(mse)
