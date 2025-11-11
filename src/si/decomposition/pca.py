import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):
    """
    Principal Component Analysis (PCA) using eigenvalue decomposition of the covariance matrix.

    Parameters
    ----------
    n_components: int
        Number of principal components to keep.
    """
    def __init__(self, n_components: int, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def _fit(self, dataset: Dataset) -> 'PCA':
        """
        Fit the model by computing the mean, principal components, and explained variance.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset.

        Returns
        -------
        self: PCA
            The fitted PCA instance.
        """
        X = dataset.X

        # 1. Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Compute the covariance matrix and perform eigenvalue decomposition
        covariance_matrix = np.cov(X_centered, rowvar=False)  # rowvar=False: columns are features
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # 3. Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 4. Store the principal components and explained variance
        self.components = eigenvectors.T[:self.n_components]  # Rows are principal components
        self.explained_variance = eigenvalues[:self.n_components]  # Eigenvalues for the selected components

        # 5. Calculate explained variance ratio (This is an extra)
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance

        return self


    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset using the principal components.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset.

        Returns
        -------
        dataset: Dataset
            The transformed dataset with reduced dimensions.
        """
        if self.components is None:
            raise RuntimeError("Fit the transformer before transforming.")

        X = dataset.X
        X_centered = X - self.mean

        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.components.T)

        # Create a new Dataset with the transformed features
        transformed_features = [f"PC{i+1}" for i in range(self.n_components)]

        return Dataset(
            X=X_transformed,
            y=dataset.y,
            features=transformed_features,
            label=dataset.label
        )
