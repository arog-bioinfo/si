from unittest import TestCase
from datasets import DATASETS_PATH
import os
import numpy as np
from si.io.csv_file import read_csv
from si.statistics.tanimoto_similarity import tanimoto_similarity


class TestTanimotoSimilarity(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_tanimoto_similarity_binary(self):
        # Test with binary vectors
        x = np.array([1, 0, 1, 1])
        y = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])
        our_similarity = tanimoto_similarity(x, y)

        # Using sklearn for comparison
        from sklearn.metrics import jaccard_score

        # Compute Jaccard similarity for each row in y
        sklearn_similarity = np.array([
            jaccard_score(x, y[0], zero_division=1),
            jaccard_score(x, y[1], zero_division=1)
        ])

        assert np.allclose(our_similarity, sklearn_similarity)

    def test_tanimoto_similarity_real(self):
        # Test with non-negative real-valued vectors
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        our_similarity = tanimoto_similarity(x, y)

        # Manually compute Tanimoto similarity for comparison
        # Tanimoto similarity formula: (x · y) / (||x||^2 + ||y||^2 - x · y)
        dot_product = np.dot(x, y.T)
        norm_x_sq = np.sum(x ** 2)
        norm_y_sq = np.sum(y ** 2, axis=1)
        expected_similarity = dot_product / (norm_x_sq + norm_y_sq - dot_product)

        assert np.allclose(our_similarity, expected_similarity)
