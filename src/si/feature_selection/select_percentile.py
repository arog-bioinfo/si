from typing import Callable
import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

#Ex.3
class SelectPercentile(Transformer):
    """
    Select features based on a percentile of the highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function.

    Parameters
    ----------
    score_func: callable, default=f_classification
        Function taking a dataset and returning a pair of arrays (scores, p_values).
    percentile: int, default=10
        Percentile of features to select (e.g., selects the top __% of features).

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, percentile: int = 10, **kwargs):
        """
        Initialize SelectPercentile.
        Parameters
        ----------
        score_func: callable, default=f_classification
            Function to compute scores and p-values.
        percentile: int, default=10
            Percentile of features to select.
        """
        super().__init__(**kwargs)
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Fit SelectPercentile to compute the F scores and p-values.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset.
        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset by selecting features based on the percentile threshold.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset.
        Returns
        -------
        dataset: Dataset
            A labeled dataset with the selected features.
        """
        if self.F is None:
            raise RuntimeError("Fit the transformer before transforming.")

        # Calculate the threshold
        threshold = np.percentile(self.F, 100 - self.percentile)

        # Create a mask for features with F-values >= threshold
        mask = self.F > threshold

        # Handle ties: include enough features to meet the percentile
        n_features = dataset.X.shape[1]
        n_selected = int(n_features * self.percentile / 100)
        if mask.sum() < n_selected:
            # Include tied features at the threshold
            tied_indices = np.where(self.F == threshold)[0]
            n_additional = n_selected - mask.sum()
            mask[tied_indices[:n_additional - 1]] = True

        # Select features
        idxs = np.where(mask)[0]
        features = np.array(dataset.features)[idxs]

        return Dataset(
            X=dataset.X[:, idxs],
            y=dataset.y,
            features=list(features),
            label=dataset.label
        )

