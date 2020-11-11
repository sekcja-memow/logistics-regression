import numpy as np
from .helpers import sigmoid, sign
from .optimizers import OptimizerInterface


class LogisticRegression:
    """
    Logistic Regression simple classifier.
    Parameters
    ----------
    optimizer: Optimizer object
        Optimizer algorithm.
        - Gradient descent optimizer with momentum.
        - Momentum Gradient descent.
    num_iterations: int, default=100
        Number of iterations for the optimizer algorithm.
    penalty: {'l1', 'l2', 'elasticnet', 'none'}, default='l1'
        Penalty form used in regularization.
        - 'l1' L1 regularization.
        - 'l2' L2 regularization.
        - 'elasticnet' Elastic net regularization.
    l1_rate: float, default=0.1
        L1 regularization coefficient.
    l2_rate: float, default=0.1
        L2 regularization coefficient.
    alpha: float, default=0.8
        Proportionality coefficient in elastic net regularization.
    threshold: float, default=0.5
        Logistic regression decision threshold.
    verbose: boolean, default=False
        Verbosity during optimization.
    Attributes
    ----------
    theta: ndarray of shape (X.shape[1],)
        Coefficient of the X (features) in logistic regression.
    Examples
    --------
    >>> model = LogisticRegression()
    >>> model.fit(X, y)
    >>> y_pred = model.pred(X_test)
    >>> model.evaluate(y_test, y_pred)
    0.9275510204081632
    """

    def __init__(self, optimizer: OptimizerInterface, num_iterations=100, threshold=0.5, verbose=False):
        self.optimizer = optimizer
        self.num_iterations = num_iterations

        self.threshold = threshold
        self.verbose = verbose

    def cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Computes the cost of logistic regression.
        Parameters
        ----------
        X: np.ndarray
            Training vector.
        y: np.ndarray
            Training labels.
        theta: np.ndarray
            Model parameters.
        Returns
        -------
        float
            The cost at the point theta.
        """
        h = sigmoid(X @ theta)
        cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        return cost

    def gradient(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the cost function, which is used by optimizer algorithm.
        Parameters
        ----------
        X: np.ndarray
            Training vector.
        y: np.ndarray
            Training labels.
        theta: np.ndarray
            Model parameters.
        Returns
        -------
        np.ndarray
            The gradient of the cost function.
        """
        m = y.size # number of probs
        h = sigmoid(X @ theta)
        grad = (1 / m) * (X.T @ (h - y))
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the given data.
        Parameters
        ----------
        X: np.ndarray
            Training vector.
        y: np.ndarray
            Training labels.
        Returns
        -------
        self
        """
        self.theta = np.zeros(X.shape[1])
        momentum = np.ones_like(self.theta)

        for epoch in range(self.num_iterations):
            grad = self.gradient(X, y, self.theta)

            self.theta = self.optimizer.optimize(self.theta, grad)

            if epoch % 10 == 0 and self.verbose:
                cost = self.cost(X, y, self.theta)
                print(f"Epoch number {epoch} of {self.num_iterations}: cost={cost}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels by the given data.
        Parameters
        ----------
        X: np.ndarray
            Test vector.
        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        return np.int8(sigmoid(np.dot(X, self.theta)) >= self.threshold)


    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = len([a for a, p in zip(y_true, y_pred) if a == p and p == 1])
        FP = len([a for a, p in zip(y_true, y_pred) if a != p and p == 1])
        return TP / (TP + FP)