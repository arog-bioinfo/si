from unittest import TestCase
from datasets import DATASETS_PATH
import os
import numpy as np
from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split 

class TestStratifiedSplits(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_stratified_train_test_split(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)

        # Check the sizes of the train and test sets
        test_samples_size = int(self.dataset.X.shape[0] * 0.2)
        self.assertEqual(test.X.shape[0], test_samples_size)
        self.assertEqual(train.X.shape[0], self.dataset.X.shape[0] - test_samples_size)

        # Check that the class distribution is preserved
        unique_train, counts_train = np.unique(train.y, return_counts=True)
        unique_test, counts_test = np.unique(test.y, return_counts=True)
        unique_original, counts_original = np.unique(self.dataset.y, return_counts=True)

        # Calculate the ratios for each class
        for label in unique_original:
            original_ratio = counts_original[unique_original == label][0] / self.dataset.X.shape[0]
            train_ratio = counts_train[unique_train == label][0] / train.X.shape[0]
            test_ratio = counts_test[unique_test == label][0] / test.X.shape[0]

            # Allow a small tolerance for rounding errors
            self.assertAlmostEqual(train_ratio, original_ratio, places=1)
            self.assertAlmostEqual(test_ratio, original_ratio, places=1)

    def test_stratified_train_test_split_random_state(self):
        # Test that the same random state produces the same split
        train1, test1 = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        train2, test2 = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        train3, test3 = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)

        self.assertTrue(np.array_equal(train1.y, train2.y))
        self.assertTrue(np.array_equal(test1.y, test2.y))
        self.assertTrue(np.array_equal(train1.X, train2.X))
        self.assertTrue(np.array_equal(test1.X, test2.X))
        self.assertFalse(np.array_equal(train1.X, train3.X))
        self.assertFalse(np.array_equal(test1.X, test3.X))
