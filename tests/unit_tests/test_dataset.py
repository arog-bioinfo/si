import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())
    
    def test_dataset_dropna(self):
        X = np.array([[1, 2], [3, np.nan], [4, 5], [np.nan, 6]])
        y = np.array([0, 1, 0, 1])

        X_expected = np.array([[1, 2], [4, 5]])
        y_expected = np.array([0, 0])


        dataset = Dataset(X, y)
        dataset_expected = Dataset(X_expected, y_expected)

        dataset.dropna()

        # Check that no NaN values remain
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(len(dataset.X), 2)
        self.assertEqual(len(dataset.y), 2)

        # Use np.array_equal or np.testing.assert_array_equal for array comparison
        np.testing.assert_array_equal(dataset.X, dataset_expected.X)
        np.testing.assert_array_equal(dataset.y, dataset_expected.y)
    
    def test_fillna_value(self):
        X = np.array([[1, 2], [3, np.nan], [4, 5], [np.nan, 6]])
        y = np.array([0, 1, 0, 1])
        X_expected = np.array([[1, 2], [3, 0], [4, 5], [0, 6]])  # Expected after filling NaN with 0

        dataset = Dataset(X, y)
        dataset_expected = Dataset(X_expected,y)

        dataset.fillna(0)

        # Check that no NaN values remain
        self.assertFalse(np.isnan(dataset.X).any())

        # Optionally, check the filled values
        np.testing.assert_array_equal(dataset.X, dataset_expected.X)
        np.testing.assert_array_equal(dataset.y, dataset_expected.y)

    def test_fillna_mean(self):
        X = np.array([[1, 2], [3, np.nan], [4, 5], [np.nan, 6]])
        y = np.array([0, 1, 0, 1])
        # Calculate the expected X after filling NaN with the mean of each feature
        X_expected = np.array([[1, 2], [3, (2 + 5 + 6) / 3], [4, 5], [(1 + 3 + 4) / 3, 6]])
        dataset = Dataset(X, y)
        dataset_expected = Dataset(X_expected, y)
        dataset.fillna("mean")

        # Check that no NaN values remain
        self.assertFalse(np.isnan(dataset.X).any())

        print(dataset.X)
        # Check the filled values
        np.testing.assert_array_almost_equal(dataset.X, dataset_expected.X, decimal=6)
        np.testing.assert_array_equal(dataset.y, dataset_expected.y)

    def test_fillna_median(self):
        X = np.array([[1, 2], [3, np.nan], [4, 5], [np.nan, 6]])
        y = np.array([0, 1, 0, 1])
        # Calculate the expected X after filling NaN with the median of each feature
        X_expected = np.array([
            [1, 2],
            [3, 5],  # Median of [2, 5, 6] is 5
            [4, 5],
            [3, 6]   # Median of [1, 3, 4] is 3
        ])
        dataset = Dataset(X, y)
        dataset_expected = Dataset(X_expected, y)
        dataset.fillna("median")

        # Check that no NaN values remain
        self.assertFalse(np.isnan(dataset.X).any())

        # Check the filled values
        np.testing.assert_array_equal(dataset.X, dataset_expected.X)
        np.testing.assert_array_equal(dataset.y, dataset_expected.y)
    
    def test_remove_by_index(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        dataset = Dataset(X, y)
        dataset.remove_by_index(1)

        # Check that the sample at index 1 is removed
        self.assertEqual(len(dataset.X), 3)
        self.assertEqual(len(dataset.y), 3)
        np.testing.assert_array_equal(dataset.X, np.array([[1, 2], [5, 6], [7, 8]]))
        np.testing.assert_array_equal(dataset.y, np.array([0, 0, 1]))

    def test_remove_by_index_out_of_bounds(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        dataset = Dataset(X, y)

        with self.assertRaises(IndexError):
            dataset.remove_by_index(10)  # Index out of bounds
