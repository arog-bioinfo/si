from unittest import TestCase
import numpy as np
from si.neural_networks.optimizers import Adam

#Ex.15
class TestAdamOptimizer(TestCase):
    def setUp(self):
        self.optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        self.weights = np.array([1.0, -1.0, 0.5])
        self.grad = np.array([0.1, -0.2, 0.05])

    def test_initialization(self):
        self.assertEqual(self.optimizer.learning_rate, 0.001)
        self.assertEqual(self.optimizer.beta_1, 0.9)
        self.assertEqual(self.optimizer.beta_2, 0.999)
        self.assertEqual(self.optimizer.epsilon, 1e-8)
        self.assertIsNone(self.optimizer.m)
        self.assertIsNone(self.optimizer.v)
        self.assertEqual(self.optimizer.t, 0)

    def test_first_update(self):
        new_weights = self.optimizer.update(self.weights, self.grad)

        self.assertEqual(self.optimizer.t, 1)
        self.assertIsNotNone(self.optimizer.m)
        self.assertIsNotNone(self.optimizer.v)
        self.assertEqual(self.optimizer.m.shape, self.weights.shape)
        self.assertEqual(self.optimizer.v.shape, self.weights.shape)
        self.assertFalse(np.array_equal(new_weights, self.weights))

    def test_multiple_updates(self):
        for _ in range(3):
            self.weights = self.optimizer.update(self.weights, self.grad)

        self.assertEqual(self.optimizer.t, 3)
        self.assertTrue(np.all(self.optimizer.m != 0))
        self.assertTrue(np.all(self.optimizer.v != 0))

