from unittest import TestCase

import numpy as np

from si.neural_networks.layers import Dropout

#Ex.12
class TestDropoutLayer(TestCase):
    def setUp(self):
        np.random.seed(0)
        self.x = np.random.rand(10, 5)
        self.dropout = Dropout(probability=0.5)

    def test_forward_training_applies_mask_and_scaling(self):
        np.random.seed(42)
        y = self.dropout.forward_propagation(self.x, training=True)

        self.assertEqual(y.shape, self.x.shape)
        self.assertIsNotNone(self.dropout.mask)

        scale = 1 / (1 - self.dropout.probability)
        mask = self.dropout.mask.astype(bool)

        # dropped units -> 0
        self.assertTrue(np.all(y[~mask] == 0))

        # kept units -> scaled
        np.testing.assert_allclose(y[mask], self.x[mask] * scale, rtol=1e-7, atol=0)

    def test_forward_inference_is_identity(self):
        y = self.dropout.forward_propagation(self.x, training=False)
        np.testing.assert_array_equal(y, self.x)

    def test_backward_masks_gradients(self):
        np.random.seed(42)
        _ = self.dropout.forward_propagation(self.x, training=True)
        mask = self.dropout.mask.astype(bool)

        grad_out = np.random.rand(*self.x.shape)
        grad_in = self.dropout.backward_propagation(grad_out)

        self.assertEqual(grad_in.shape, grad_out.shape)
        self.assertTrue(np.all(grad_in[~mask] == 0))
        np.testing.assert_array_equal(grad_in[mask], grad_out[mask])

    def test_invalid_probability_raises(self):
        with self.assertRaises(ValueError):
            Dropout(-0.1)
        with self.assertRaises(ValueError):
            Dropout(1.0)

    def test_parameters_is_zero(self):
        self.assertEqual(self.dropout.parameters(), 0)
