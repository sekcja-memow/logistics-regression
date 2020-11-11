import numpy as np
from .helpers import sigmoid, sign


class LogisticRegression:
    """
    Logistic Regression simple classifier.
    Parameters
    ----------
    learning_rate: float, default=0.05
        Learning rate parameter used in optimizer function.
    optimizer: {'momentum', 'sgd'}, default='momentum'
        Optimizer algorithm.
        - 'momentum' Stochastic gradient descent optimizer with momentum.
        - 'sgd' Stochastic gradient descent optimizer.
    momentum_rate: float, default=0.9, optional
        Momentum rate parameter used in momentum optimizer algorithm.
        Momentum rate must be from interval from 0 to 1.
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

    def __init__(self, learning_rate=0.05, optimizer='momentum', momentum_rate=0.9,
                 num_iterations=100, penalty='l1', l1_rate=0.1, l2_rate=0.1, alpha=0.8,
                 threshold=0.5, verbose=False):

        self.learning_rate = learning_rate

        if optimizer not in ['momentum', 'sgd']:
            raise ValueError('Optimizer must be one of the available: {"momentum", "sgd"}')

        self.optimizer = optimizer

        if momentum_rate < 0 or momentum_rate > 1:
            raise ValueError('Momentum rate must be from interval from 0 to 1.')

        self.momentum_rate = momentum_rate
        self.num_iterations = num_iterations

        if penalty not in ['l1', 'l2', 'elasticnet', 'none']:
            raise ValueError('Penalty must be one of the available: {"l1", "l2", "none"}')

        self.penalty = penalty

        self.l1_rate = l1_rate
        self.l2_rate = l2_rate
        self.alpha = alpha

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

        if self.penalty == 'none' or self.penalty is None:
            return cost
        elif self.penalty == 'l1':
            l1 = self.l1_rate * np.sum(np.abs(theta))
            return cost + l1
        elif self.penalty == 'l2':
            l2 = 0.5 * self.l2_rate * np.power(np.sum(theta),2)
            return cost + l2
        elif self.penalty == 'elasticnet':
            l1 = self.l1_rate * np.sum(np.abs(theta))
            l2 = 0.5 * self.l1_rate * np.power(np.sum(theta),2)
            return cost + self.alpha * l1 + (1 - self.alpha)/2 * l2


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

        if self.penalty == 'none' or self.penalty is None:
            return grad
        elif self.penalty == 'l1':
            l1_grad = self.l1_rate * np.array([sign(theta_i) for theta_i in theta])
            return grad + l1_grad
        elif self.penalty == 'l2':
            l2_grad = (self.l2_rate / m) * theta
            return grad + l2_grad
        elif self.penalty == 'elasticnet':
            l1_grad = self.l1_rate * np.array([sign(theta_i) for theta_i in theta])
            l2_grad = (self.l1_rate / m) * theta
            return grad + self.alpha * l1_grad * (1-self.alpha)/2 * l2_grad

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

            if self.optimizer == 'momentum':
                momentum = self.momentum_rate * momentum - self.learning_rate * grad
                self.theta = self.theta + momentum
            else:
                self.theta = self.theta - self.learning_rate * grad

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