from unittest import TestCase
import numpy as np
from si.metrics.rmse import rmse

#Ex.7
class TestRMSE(TestCase):
    def test_rmse_perfect_prediction(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        self.assertEqual(rmse(y_true, y_pred), 0.0)

    def test_rmse_simple_case(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0.5, 1.5, 2.5, 3.5])
        expected_rmse = np.sqrt(0.25)  # sqrt(mean([0.25, 0.25, 0.25, 0.25]))
        self.assertAlmostEqual(rmse(y_true, y_pred), expected_rmse)

    def test_rmse_large_values(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 210, 310])
        expected_rmse = np.sqrt(100)  # sqrt(mean([100, 100, 100]))
        self.assertAlmostEqual(rmse(y_true, y_pred), expected_rmse)