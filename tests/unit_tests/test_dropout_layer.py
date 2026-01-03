from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from datasets import DATASETS_PATH

import os

from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.neural_networks.layers import Dropout
from si.neural_networks.optimizers import Optimizer

#Ex.12
class TestDropoutLayer(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_forward_propagation_training(self):
        # Test forward propagation in training mode
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape((self.dataset.X.shape[1],))

        # Save random state for reproducibility
        rng_state = np.random.get_state()
        np.random.seed(42)  # Set seed for reproducible test

        output = dropout.forward_propagation(self.dataset.X, training=True)

        # Verify output shape
        self.assertEqual(output.shape[0], self.dataset.X.shape[0])
        self.assertEqual(output.shape[1], self.dataset.X.shape[1])

        # Verify that some values are zero (due to dropout)
        self.assertTrue(np.any(output == 0))

        # Verify scaling factor is applied
        non_zero_values = output[output != 0]
        self.assertTrue(np.all(non_zero_values > 1))  # Values should be scaled up

        # Restore random state
        np.random.set_state(rng_state)

    def test_forward_propagation_inference(self):
        # Test forward propagation in inference mode
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape((self.dataset.X.shape[1],))

        output = dropout.forward_propagation(self.dataset.X, training=False)

        # Verify output is identical to input in inference mode
        np.testing.assert_array_equal(output, self.dataset.X)

    def test_backward_propagation(self):
        # Test backward propagation
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape((self.dataset.X.shape[1],))

        # Save random state for reproducibility
        rng_state = np.random.get_state()
        np.random.seed(42)  # Set seed for reproducible test

        # First do forward pass to create mask
        dropout.forward_propagation(self.dataset.X, training=True)

        # Create random output error
        output_error = np.random.random(self.dataset.X.shape)

        # Perform backward propagation
        input_error = dropout.backward_propagation(output_error)

        # Verify input error shape
        self.assertEqual(input_error.shape[0], self.dataset.X.shape[0])
        self.assertEqual(input_error.shape[1], self.dataset.X.shape[1])

        # Verify that the error is masked (some values should be zero)
        self.assertTrue(np.any(input_error == 0))

        # Restore random state
        np.random.set_state(rng_state)

    def test_output_shape(self):
        # Test output shape method
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape((self.dataset.X.shape[1],))

        output_shape = dropout.output_shape()
        self.assertEqual(output_shape, (self.dataset.X.shape[1],))

    def test_parameters(self):
        # Test parameters method
        dropout = Dropout(probability=0.5)
        self.assertEqual(dropout.parameters(), 0)

    def test_invalid_probability(self):
        # Test that invalid probability raises ValueError
        with self.assertRaises(ValueError):
            Dropout(probability=-0.1)

        with self.assertRaises(ValueError):
            Dropout(probability=1.1)

class TestDropoutLayerRN(TestCase):
    def setUp(self):
        # Create a fixed random input for consistent testing
        np.random.seed(42)
        self.input = np.random.rand(10, 5)  # 10 samples, 5 features
        self.dropout = Dropout(probability=0.5)

    def test_training_mode_behavior(self):
        """Test dropout behavior in training mode with random input"""
        # Save random state for reproducibility
        rng_state = np.random.get_state()
        np.random.seed(42)  # Set seed for reproducible test

        # Forward pass in training mode
        output = self.dropout.forward_propagation(self.input, training=True)

        # Check output shape matches input shape
        self.assertEqual(output.shape, self.input.shape)

        # Check that some values are zero (dropout applied)
        self.assertTrue(np.any(output == 0))

        # Check that non-zero values are scaled by 1/(1-p)
        non_zero_mask = output != 0
        expected_scale = 1 / (1 - self.dropout.probability)
        np.testing.assert_allclose(
            output[non_zero_mask],
            self.input[non_zero_mask] * expected_scale,
            rtol=1e-5
        )

        # Check that the mask was stored
        self.assertIsNotNone(self.dropout.mask)
        self.assertEqual(self.dropout.mask.shape, self.input.shape)

        # Check that the mask contains both 0s and 1s
        self.assertTrue(np.any(self.dropout.mask == 0))
        self.assertTrue(np.any(self.dropout.mask == 1))

        # Restore random state
        np.random.set_state(rng_state)

    def test_inference_mode_behavior(self):
        """Test dropout behavior in inference mode with random input"""
        # Forward pass in inference mode
        output = self.dropout.forward_propagation(self.input, training=False)

        # Check output is identical to input
        np.testing.assert_array_equal(output, self.input)

        # Check that no mask was created
        self.assertIsNone(self.dropout.mask)

    def test_backward_propagation_behavior(self):
        """Test backward propagation behavior with random error"""
        # Save random state for reproducibility
        rng_state = np.random.get_state()
        np.random.seed(42)  # Set seed for reproducible test

        # First do forward pass to create mask
        self.dropout.forward_propagation(self.input, training=True)
        saved_mask = self.dropout.mask.copy()

        # Create random output error
        output_error = np.random.rand(*self.input.shape)

        # Perform backward propagation
        input_error = self.dropout.backward_propagation(output_error)

        # Check that error was masked correctly
        np.testing.assert_array_equal(
            input_error * (1 - saved_mask),
            np.zeros_like(input_error)
        )

        # Check that non-masked values remain unchanged
        np.testing.assert_array_equal(
            input_error * saved_mask,
            output_error * saved_mask
        )

        # Restore random state
        np.random.set_state(rng_state)

    def test_dropout_rate(self):
        """Test that the dropout rate is approximately correct"""
        # Save random state for reproducibility
        rng_state = np.random.get_state()
        np.random.seed(42)  # Set seed for reproducible test

        # Run many trials to check dropout rate
        n_trials = 10000
        large_input = np.ones((n_trials, 1))
        self.dropout.forward_propagation(large_input, training=True)

        # Calculate actual dropout rate
        actual_rate = np.mean(self.dropout.mask == 0)
        expected_rate = self.dropout.probability

        # Check that actual rate is close to expected rate
        np.testing.assert_allclose(actual_rate, expected_rate, atol=0.02)

        # Restore random state
        np.random.set_state(rng_state)