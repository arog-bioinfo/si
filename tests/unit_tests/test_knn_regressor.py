from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.models.knn_regressor import KNNRegressor
from si.model_selection.split import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

class TestKNNRegressor(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNRegressor(k=3)
        knn.fit(self.dataset)
        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        knn = KNNRegressor(k=1)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)
        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        self.assertEqual(predictions.shape, test_dataset.y.shape)

    def test_score(self):
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)
        knn.fit(train_dataset)
        score = knn.score(test_dataset)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_sklearn_comparison(self):
        # Split the dataset
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)

        # Fit and predict with implemented KNNRegressor
        knn = KNNRegressor(k=3)
        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        rmse = knn.score(test_dataset)

        # Fit and predict with scikit-learn's KNeighborsRegressor
        sklearn_knn = KNeighborsRegressor(n_neighbors=3)
        sklearn_knn.fit(train_dataset.X, train_dataset.y)
        sklearn_predictions = sklearn_knn.predict(test_dataset.X)
        sklearn_rmse = np.sqrt(mean_squared_error(test_dataset.y, sklearn_predictions))

        # Compare RMSE scores
        self.assertGreaterEqual(rmse, sklearn_rmse)