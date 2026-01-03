from unittest import TestCase
import os
import numpy as np
from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.metrics.accuracy import accuracy
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression


class TestRandomizedSearchCV(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_randomized_search_cv(self):
        # Create LogisticRegression model
        model = LogisticRegression()

        # Define hyperparameter distributions
        parameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10).tolist(),  # 10 equal intervals between 1 and 10
            'alpha': np.linspace(0.001, 0.0001, 100).tolist(),  # 100 equal intervals between 0.001 and 0.0001
            'max_iter': np.linspace(1000, 2000, 200).tolist()  # 200 equal intervals between 1000 and 2000
        }

        # Perform randomized search with n_iter=10 and cv=3
        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=parameter_grid,
            scoring=accuracy,
            cv=3,
            n_iter=10
        )

        # Verify the results structure
        self.assertEqual(len(results["scores"]), 10)  # Should have 10 scores for 10 iterations
        self.assertEqual(len(results["hyperparameters"]), 10)  # Should have 10 hyperparameter sets

        # Verify each hyperparameter set has 3 parameters
        for params in results["hyperparameters"]:
            self.assertEqual(len(params), 3)
            self.assertIn("l2_penalty", params)
            self.assertIn("alpha", params)
            self.assertIn("max_iter", params)

        # Optional Printing
        # print("\nRandomized Search Results:")
        # print(f"Number of iterations: {len(results['scores'])}")
        # print(f"All scores: {[round(score, 4) for score in results['scores']]}")
        # print(f"Best hyperparameters: {results['best_hyperparameters']}")
        # print(f"Best score: {round(results['best_score'], 4)}")

        # Verify best hyperparameters structure
        best_hyperparameters = results['best_hyperparameters']
        self.assertEqual(len(best_hyperparameters), 3)
        self.assertIn("l2_penalty", best_hyperparameters)
        self.assertIn("alpha", best_hyperparameters)
        self.assertIn("max_iter", best_hyperparameters)

        # Verify best score is reasonable (should be between 0 and 1)
        best_score = results['best_score']
        self.assertGreaterEqual(best_score, 0)
        self.assertLessEqual(best_score, 1)

        # Check that the best score is actually the maximum score
        self.assertEqual(best_score, max(results['scores']))

        # Optional Printing
        # print(f"\nBest score (rounded): {round(best_score, 2)}")