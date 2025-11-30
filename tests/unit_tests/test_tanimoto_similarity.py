from unittest import TestCase
import numpy as np
from sklearn.metrics import jaccard_score
from si.statistics.tanimoto_similarity import tanimoto_similarity

class TestTanimotoSimilarity(TestCase):
    def setUp(self):
        """Set up test fixtures with random binary data."""
        # Create a larger random binary dataset
        np.random.seed(42)  # For reproducibility
        self.n_samples = 10
        self.n_features = 5

        # Generate random binary vectors (0s and 1s)
        self.x = np.random.randint(0, 2, size=self.n_features)
        self.y = np.random.randint(0, 2, size=(self.n_samples, self.n_features))


    def test_tanimoto_similarity_binary(self):
        """Test Tanimoto similarity with random binary vectors."""
        # Calculate our implementation's similarity
        our_similarity = tanimoto_similarity(self.x, self.y)

        # Calculate sklearn's Jaccard similarity for comparison
        sklearn_similarity = np.array([
            jaccard_score(self.x, y_row)
            for y_row in self.y
        ])

        # Verify the results match
        np.testing.assert_array_equal(our_similarity, sklearn_similarity)

    def test_tanimoto_similarity_edge_cases(self):
        """Test edge cases: all zeros, all ones, and mixed vectors."""
        # Test with all zeros
        x_zeros = np.zeros(self.n_features)
        y_zeros = np.zeros((5, self.n_features))
        sim_zeros = tanimoto_similarity(x_zeros, y_zeros)
        self.assertTrue(np.all(sim_zeros == 1.0))  # All zeros should have similarity 1

        # Test with all ones
        x_ones = np.ones(self.n_features)
        y_ones = np.ones((5, self.n_features))
        sim_ones = tanimoto_similarity(x_ones, y_ones)
        self.assertTrue(np.all(sim_ones == 1.0))  # All ones should have similarity 1

        # Test with one zero vector and one one vector
        x_mixed = np.ones(self.n_features)
        x_mixed[0] = 0
        y_mixed = np.zeros(self.n_features)
        y_mixed[0] = 1
        sim_mixed = tanimoto_similarity(x_mixed, y_mixed.reshape(1, -1))
        self.assertEqual(sim_mixed[0], 0.0)  # Completely different should have similarity 0
