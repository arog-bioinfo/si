from unittest import TestCase

from datasets import DATASETS_PATH

import os
import numpy as np
from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression

#Ex.10
class TestStackingClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):

        decision_tree = DecisionTreeClassifier()
        knn = KNNClassifier(k=2)
        logistic_regression = LogisticRegression()
        knn_f = KNNClassifier(k=2)

        vc=StackingClassifier(models = [decision_tree, knn, logistic_regression], final_model= knn_f)

        vc.fit(self.train_dataset)
        
        self.assertEqual(vc.models[0].min_sample_split, 2)
        self.assertEqual(vc.models[0].max_depth, 10)


        self.assertTrue(np.all(self.train_dataset.features == vc.models[1].dataset.features))
        self.assertTrue(np.all(self.train_dataset.y == vc.models[1].dataset.y))


        self.assertEqual(vc.models[2].theta.shape[0], self.train_dataset.shape()[1])
        self.assertNotEqual(vc.models[2].theta_zero, None)
        self.assertNotEqual(len(vc.models[2].cost_history), 0)
        self.assertNotEqual(len(vc.models[2].mean), 0)
        self.assertNotEqual(len(vc.models[2].std), 0)


    def test_predict(self):
        decision_tree = DecisionTreeClassifier()
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        knn_f = KNNClassifier()

        vc=StackingClassifier(models = [decision_tree, knn, logistic_regression], final_model= knn_f)

        vc.fit(self.train_dataset)

        predictions = vc.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])
    
    def test_score(self):
        decision_tree = DecisionTreeClassifier()
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        knn_f = KNNClassifier()

        vc=StackingClassifier(models = [decision_tree, knn, logistic_regression], final_model= knn_f)
        vc.fit(self.train_dataset)
        accuracy_ = vc.score(self.test_dataset)
        
        print(accuracy_)
        self.assertEqual(round(accuracy_, 2), 0.95)