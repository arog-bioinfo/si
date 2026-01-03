import numpy as np
from typing import List
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

#Ex.10
class StackingClassifier(Model):
    def __init__(self, models: List[Model], final_model: Model):
        """
        Initialize the StackingClassifier.

        Parameters
        ----------
        models : List[Model]
            List of base models for the first level of stacking.
        final_model : Model
            The final model to make predictions based on the base models' outputs.
        """
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the stacking classifier to the training data.

        Parameters
        ----------
        dataset : Dataset
            The training dataset with features and labels.

        Returns
        -------
        self : StackingClassifier
            The fitted stacking classifier.
        """
        # Step 1: Train the initial set of models
        for model in self.models:
            model.fit(dataset)

        # Step 2: Get predictions from the initial set of models
        # Create a new dataset with the predictions as features
        predictions = []
        for model in self.models:
            pred = model.predict(dataset)
            predictions.append(pred.reshape(-1, 1))  # Reshape to 2D array

        # Combine predictions horizontally
        stacked_features = np.hstack(predictions)

        # Step 3: Train the final model with the predictions of the initial set of models
        stacked_dataset = Dataset(X=stacked_features, y=dataset.y)
        self.final_model.fit(stacked_dataset)

        # Step 4: Return itself
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels using the stacking classifier.

        Parameters
        ----------
        dataset : Dataset
            Input data to predict.

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels.
        """
        # Step 1: Get predictions from the initial set of models
        predictions = []
        for model in self.models:
            pred = model.predict(dataset)
            predictions.append(pred.reshape(-1, 1))  # Reshape to 2D array

        # Combine predictions horizontally
        stacked_features = np.hstack(predictions)

        # Step 2: Get the final predictions using the final model
        final_predictions = self.final_model.predict(Dataset(X=stacked_features, y=None))

        return final_predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.
        predictions: np.ndarray
            Predictions

        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, predictions)