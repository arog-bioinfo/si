import numpy as np
from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification

#Ex.3
class TestSelectPercentile(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        select_percentile = SelectPercentile(score_func=f_classification, percentile=50)
        select_percentile.fit(self.dataset)
        self.assertTrue(select_percentile.F.shape[0] > 0)
        self.assertTrue(select_percentile.p.shape[0] > 0)

    def test_transform(self):
        select_percentile = SelectPercentile(score_func=f_classification, percentile=50)
        select_percentile.fit(self.dataset)
        new_dataset = select_percentile.transform(self.dataset)

        #check that the number of features is the expected percentile
        expected_features = int(len(self.dataset.features) * 0.5)  # 50% of features
        self.assertEqual(len(new_dataset.features), expected_features)
