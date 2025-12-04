from unittest import TestCase
import os
import numpy as np
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

#Ex.5
class TestPCA(TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.iris_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.iris_dataset = read_csv(filename=self.iris_file, features=True, label=True)

        # Initialize and fit both PCA implementations
        self.pca = PCA(n_components=2)
        self.pca.fit(self.iris_dataset)

        # Prepare data for scikit-learn's PCA
        self.scaler = StandardScaler(with_std=False)
        self.X_scaled = self.scaler.fit_transform(self.iris_dataset.X)
        self.sklearn_pca = SklearnPCA(n_components=2)
        self.sklearn_pca.fit(self.X_scaled)

        # Transform data with both implementations
        self.transformed_data = self.pca.transform(self.iris_dataset)
        self.sklearn_transformed = self.sklearn_pca.transform(self.X_scaled)

    def test_components_shape(self):
        """Test that components have the correct shape."""
        self.assertEqual(self.pca.components.shape, (2, self.iris_dataset.X.shape[1]))
        self.assertEqual(self.sklearn_pca.components_.shape, (2, self.iris_dataset.X.shape[1]))

    def test_explained_variance(self):
        """Test that explained variance is similar between implementations."""
        # Compare explained variance ratio
        np.testing.assert_allclose(
            self.pca.explained_variance_ratio,
            self.sklearn_pca.explained_variance_ratio_,
            rtol=1e-3
        )

    def test_transform_shape(self):
        """Test that transformed data has the correct shape."""
        self.assertEqual(self.transformed_data.X.shape, (self.iris_dataset.X.shape[0], 2))
        self.assertEqual(self.sklearn_transformed.shape, (self.iris_dataset.X.shape[0], 2))

    def test_components_alignment(self):
        """Test that principal components are aligned (up to sign flips)."""

        for i in range(2):
            component_match = np.allclose(
                np.abs(self.pca.components[i]),
                np.abs(self.sklearn_pca.components_[i]),
                rtol=1e-3
            )
            self.assertTrue(component_match)

    def test_transformed_data(self):
        """Test that transformed data is similar (up to sign flips)."""
        np.testing.assert_allclose(
            np.abs(self.transformed_data.X),
            np.abs(self.sklearn_transformed),
            rtol=1e-3
        )