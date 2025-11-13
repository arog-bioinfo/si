from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class TestRidgeRegressionLeastSquares(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        """Test that the model fits correctly and stores the expected attributes."""
        ridge = RidgeRegressionLeastSquares()
        ridge.fit(self.train_dataset)

        # Check that theta has the correct shape
        self.assertEqual(ridge.theta.shape[0], self.train_dataset.X.shape[1])
        # Check that theta_zero is not None
        self.assertIsNotNone(ridge.theta_zero)

        # Check that mean and std are computed if scale=True
        if ridge.scale:
            self.assertIsNotNone(ridge.mean)
            self.assertIsNotNone(ridge.std)
            self.assertEqual(len(ridge.mean), self.train_dataset.X.shape[1])
            self.assertEqual(len(ridge.std), self.train_dataset.X.shape[1])

    def test_predict(self):
        """Test that the model makes predictions of the correct shape."""
        ridge = RidgeRegressionLeastSquares()
        ridge.fit(self.train_dataset)
        predictions = ridge.predict(self.test_dataset)

        # Check that predictions have the correct shape
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])

    def test_score(self):
        """Test that the model's score method returns a valid MSE."""
        ridge = RidgeRegressionLeastSquares()
        ridge.fit(self.train_dataset)
        mse = ridge.score(self.test_dataset)

        # Check that MSE is a non-negative float
        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0)

    def test_sklearn_comparison(self):
        """Test that the model's results match scikit-learn's Ridge implementation."""
        # Fit with our implementation
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge.fit(self.train_dataset)
        our_mse = ridge.score(self.test_dataset)

        # Compare with scikit-learn's Ridge
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.train_dataset.X)
        X_test_scaled = scaler.transform(self.test_dataset.X)

        sklearn_ridge = Ridge(alpha=1.0)
        sklearn_ridge.fit(X_train_scaled, self.train_dataset.y)
        sklearn_predictions = sklearn_ridge.predict(X_test_scaled)
        sklearn_mse = mean_squared_error(self.test_dataset.y, sklearn_predictions)

        # Assert that MSE values are close
        self.assertAlmostEqual(our_mse, sklearn_mse, places=4)

        # Assert that coefficients are close
        np.testing.assert_allclose(ridge.theta, sklearn_ridge.coef_, rtol=1e-4)
        self.assertAlmostEqual(ridge.theta_zero, sklearn_ridge.intercept_, places=4)

