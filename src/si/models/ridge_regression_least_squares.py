import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

#Ex.8
class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression using the least squares (closed-form) solution.
    This model solves the linear regression problem with L2 regularization using the normal equation.

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter.
    scale: bool
        Whether to scale the dataset or not.

    Attributes
    ----------
    theta: np.ndarray
        The model parameters, namely the coefficients of the linear model.
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
    mean: np.ndarray
        The mean of the dataset (for every feature).
    std: np.ndarray
        The standard deviation of the dataset (for every feature).
    """
    def __init__(self, l2_penalty: float = 1, scale: bool = True, **kwargs):
        """
        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter.
        scale: bool
            Whether to scale the dataset or not.
        """
        # parameters
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale

        # attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the model to the dataset using the closed-form solution.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model.
        """
        if self.scale:
            # Compute mean and std
            self.mean = np.mean(dataset.X, axis=0)
            self.std = np.std(dataset.X, axis=0)

            # Scale the dataset
            X = (dataset.X - self.mean) / self.std

        else:
            X = dataset.X

        m, n = X.shape

        # Add a column of ones to X for the intercept term
        X_with_intercept = np.c_[np.ones(m), X]

        # Compute the regularization matrix
        regularization = self.l2_penalty * np.eye(n + 1)
        regularization[0, 0] = 0  # Don't regularize the intercept term

        # Compute the coefficients using the normal equation for ridge regression in steps
        # Compute X^T * X (covariance matrix)
        covariance_matrix = np.dot(X_with_intercept.T, X_with_intercept)

        # Add regularization term (λI)
        regularized_covariance = covariance_matrix + regularization

        # Compute the inverse of the regularized covariance matrix
        covariance_inverse = np.linalg.inv(regularized_covariance)

        # Compute X^T * y
        xt_y = np.dot(X_with_intercept.T, dataset.y)

        # Solve for θ: (X^T X + λI)^-1 X^T y
        theta = np.dot(covariance_inverse, xt_y)
        
        # Separate the intercept and coefficients
        self.theta_zero = theta[0]
        self.theta = theta[1:]
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the output of the dataset.
        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of.
        Returns
        -------
        predictions: np.ndarray
            The predictions of the dataset.
        """
        # 1. Scale the data if required
        if self.scale:
            X = (dataset.X - self.mean) / self.std  
        else: 
            X = dataset.X

        # 2. Add intercept term to X (column of 1s as the first column)
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]  # Add column of 1s

        # 3. Concatenate theta_zero and theta into a single vector
        thetas = np.r_[self.theta_zero, self.theta]  # Combine intercept and coefficients

        # 4. Compute the predicted Y (X * thetas)
        return np.dot(X_with_intercept, thetas)


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the Mean Square Error of the model on the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on.

        predictions: np.ndarray
            Predictions

        Returns
        -------
        mse: float
            The Mean Square Error of the model.
        """
        predictions = self.predict(dataset)
        return mse(dataset.y, predictions)