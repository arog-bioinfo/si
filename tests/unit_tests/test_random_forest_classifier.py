from unittest import TestCase
import os
from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.random_forest_classifier import RandomForestClassifier
from si.models.decision_tree_classifier import DecisionTreeClassifier

#Ex.9
class TestRandomForestClassifier(TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        """Test that the random forest fits correctly with the specified number of trees."""
        n_trees = 15
        depth = 13
        split = 3
        random_forest = RandomForestClassifier(n_estimators = n_trees, max_depth = depth, min_sample_split = split)
        random_forest.fit(self.train_dataset)

        # Check that the forest was created with the correct number of trees
        self.assertEqual(len(random_forest.trees), n_trees)

        # Check that each tree has the expected default parameters
        for _, tree in random_forest.trees:
            self.assertEqual(tree.min_sample_split, 3)
            self.assertEqual(tree.max_depth, 13)

    def test_predict(self):
        """Test that the random forest makes predictions of the correct shape."""
        random_forest = RandomForestClassifier()
        random_forest.fit(self.train_dataset)
        predictions = random_forest.predict(self.test_dataset)

        # Check that predictions have the correct shape
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])

    def test_score(self):
        """Test that the random forest's score method returns a valid accuracy."""
        random_forest = RandomForestClassifier()
        random_forest.fit(self.train_dataset)
        accuracy = random_forest.score(self.test_dataset)

        # Check that accuracy is a float between 0 and 1
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

        # For breast cancer dataset, random forest should perform reasonably well
        self.assertGreaterEqual(accuracy, 0.90)

    def test_comparison_with_decision_tree(self):
        """Test that the random forest performs at least as well as a single decision tree."""
        # Train both models
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.train_dataset)
        dt_accuracy = decision_tree.score(self.test_dataset)

        random_forest = RandomForestClassifier(seed=42)
        random_forest.fit(self.train_dataset)
        rf_accuracy = random_forest.score(self.test_dataset)

        # Random forest should perform at least as well as a single decision tree
        self.assertGreaterEqual(rf_accuracy, dt_accuracy)

        # Print accuracies for comparison (optional)
        # print(f"\nDecision Tree Accuracy: {dt_accuracy:.4f}")
        # print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
