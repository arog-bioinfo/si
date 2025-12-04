import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy

#Ex.9
class RandomForestClassifier(Model):
    """
    Random Forest Classifier that fits multiple decision trees on bootstrapped samples.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees in the forest.
    max_features : int or None, default=None
        Number of features to consider for each split.
        If None, uses sqrt(n_features).
    min_sample_split : int, default=2
        Minimum samples required to split a node.
    max_depth : int, default=10
        Maximum depth of each tree.
    mode : str, default='gini'
        Split quality measure ('gini' or 'entropy').
    seed : int or None, default=None
        Random seed for reproducibility.
    """


    def __init__(self, n_estimators: int = 10, max_features: int = None,
                 min_sample_split: int = 2, max_depth: int = 10,
                 mode: str = 'gini', seed: int = None, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        self.trees = [] # Stores (feature_indices, tree) tuples
        self.feature_indices = [] # Stores unique class labels

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Fit random forest to training data using bootstrap samples.

        Parameters
        ----------
        dataset : Dataset
            Training data with features and labels.

        Returns
        -------
        self : RandomForestClassifier
            Fitted model.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        n_samples, n_features = dataset.X.shape
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            # Create bootstrap sample
            sample_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = dataset.X[sample_idx]
            y_sample = dataset.y[sample_idx]

            # Select random features
            feature_idx = np.random.choice(n_features, self.max_features, replace=False)
            X_bootstrap = X_sample[:, feature_idx]
            bootstrap_dataset = Dataset(X=X_bootstrap, y=y_sample)

            # Train and store tree
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(bootstrap_dataset)
            self.trees.append((feature_idx, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels using majority voting across all trees.

        Parameters
        ----------
        dataset : Dataset
            Input data to predict.

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels.
        """
        # Collect predictions from all trees
        tree_predictions = []
        for feature_idx, tree in self.trees:
            X_subset = dataset.X[:, feature_idx]
            subset_dataset = Dataset(X=X_subset, y=None)
            tree_predictions.append(tree.predict(subset_dataset))

        # Majority vote
        all_predictions = np.array(tree_predictions).T
        predictions = []
        for sample_preds in all_predictions:
            unique, counts = np.unique(sample_preds, return_counts=True)
            predictions.append(unique[np.argmax(counts)])

        return np.array(predictions)


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate model accuracy on given dataset.

        Parameters
        ----------
        dataset : Dataset
            Data to evaluate.

        Returns
        -------
        accuracy : float
            Fraction of correct predictions.
        """

        return accuracy(dataset.y, predictions)
