from typing import Callable
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor(Model):
    """
    KNN Regressor
    The k-Nearest Neighbors regressor is a machine learning model that predicts continuous target values
    based on a similarity measure (e.g., distance functions). This algorithm predicts the target value
    of new samples by averaging the values of the k-nearest samples in the training data.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use.
    distance: Callable
        The distance function to use.
    """
    def __init__(self, k: int = 5, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN regressor.

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use.
        distance: Callable
            The distance function to use.
        """
        # parameters
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Fit the model to the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        self: KNNRegressor
            The fitted model.
        """
        self.dataset = dataset
        return self

    def _get_prediction(self, sample: np.ndarray) -> float:
        """
        Predict the target value for the given sample.

        Parameters
        ----------
        sample: np.ndarray
            The sample to predict the target value for.

        Returns
        -------
        prediction: float
            The predicted target value.
        """
        # Compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        
        # Get the target values of the k nearest neighbors
        k_nearest_neighbors_values = self.dataset.y[k_nearest_neighbors]

        # Return the average of the k nearest neighbors' values
        return np.mean(k_nearest_neighbors_values)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the target values for the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        predictions = np.apply_along_axis(self._get_prediction, axis=1, arr=dataset.X)
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the RMSE of the model on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on.

        Returns
        -------
        rmse: float
            The root mean squared error of the model.
        """

        return rmse(dataset.y, predictions)
