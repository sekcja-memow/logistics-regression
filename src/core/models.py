import numpy as np
from sklearn.utils import shuffle
from .helpers import sigmoid
from .optimizers import OptimizerInterface, AdamOptimizer
from .regularizers import RegularizerInterface, RidgeRegularizer


class LogisticRegression:
    """
    Logistic Regression simple classifier.
    Parameters
    ----------
    optimizer: Optimizer instance: OptimizerInterface, default=AdamOptimizer()
        Optimizer algorithm used to optimize model cost.
    regularizer: Regularizer instance: RegularizerInterface, default=RidgeRegularizer()
        Regularization algorithm is used to limit the models cost function.
    num_iterations: int, default=300
        Number of iterations for the optimizer algorithm..
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
    theta: np.ndarray

    def __init__(self, optimizer: OptimizerInterface = AdamOptimizer(),
                 regularizer: RegularizerInterface = RidgeRegularizer(),
                 num_iterations=300, threshold: np.float = 0.5, fit_intercept: bool = True, verbose: bool = False):
        self.optimizer = optimizer
        self.regularizer = regularizer
        if num_iterations <= 0:
            raise ValueError('num_iterations must be greater than 0')
        self.num_iterations = num_iterations
        if threshold < 0 or threshold > 1:
            raise ValueError('Threshold value must be between 0 and 1')
        self.threshold = threshold
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def add_intercept(self, X) -> np.ndarray:
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def cost(self, X: np.ndarray, y: np.ndarray) -> float:
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
        h = sigmoid(X @ self.theta)
        cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        if self.regularizer:
            return cost + self.regularizer.cost(self.theta)
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
        m = y.size
        h = sigmoid(X @ theta)
        grad = (1 / m) * (X.T @ (h - y))
        if self.regularizer:
            return grad + self.regularizer.gradient(self.theta)
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
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
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.theta = np.zeros(X.shape[1])

        for epoch in range(self.num_iterations):
            grad = self.gradient(X, y, self.theta)

            self.theta = self.optimizer.optimize(self.theta, grad)

            if epoch % 10 == 0 and self.verbose:
                cost = self.cost(X, y)
                print(
                    f"Epoch number {epoch} of {self.num_iterations}: loss:{cost}")
        return self

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
        if self.fit_intercept:
            X = self.add_intercept(X)
        return np.int8(sigmoid(np.dot(X, self.theta)) >= self.threshold)

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = len([a for a, p in zip(y_true, y_pred) if a == p and p == 1])
        FP = len([a for a, p in zip(y_true, y_pred) if a != p and p == 1])
        return TP / (TP + FP)


class SVM:
    def __init__(self):
        self.max_epochs = 1024
        self.regularization_strength = 10000
        self.learning_rate = 0.000001
        self.cost_threshold = 0.01

        self.weights = []

    def cost(self, W, X, Y):
        # calculate hinge loss
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = self.regularization_strength * (np.sum(distances) / N)

        # calculate cost
        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost

    def cost_gradient(self, W, X_batch, Y_batch):
        # if only one example is passed (eg. in case of SGD)
        if type(Y_batch) == np.float64:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])  # gives multidimensional array

        distance = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))

        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.regularization_strength *
                          Y_batch[ind] * X_batch[ind])
            dw += di

        dw = dw/len(Y_batch)  # average
        return dw

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Add b in form of intercept
        samples, features = X.shape
        X_intercept = np.ones((samples, features + 1))
        X_intercept[:, :-1] = X
        # Temp variables
        weights = np.zeros(features + 1)
        nth = 0
        prev_cost = float("inf")
        # stochastic gradient descent
        for epoch in range(1, self.max_epochs):
            # shuffle to prevent repeating update cycles
            Xs, ys = shuffle(X_intercept, y)
            for ind, x in enumerate(Xs):
                ascent = self.cost_gradient(weights, x, ys[ind])
                weights = weights - (self.learning_rate * ascent)

            # convergence check on 2^nth epoch
            if epoch == 2 ** nth or epoch == self.max_epochs - 1:
                cost = self.cost(weights, X_intercept, y)
                print("Epoch is: {} and Cost is: {}".format(epoch, cost))
                # stoppage criterion
                if abs(prev_cost - cost) < self.cost_threshold * prev_cost:
                    self.weights = weights
                    return
                prev_cost = cost
                nth += 1
        self.weights = weights

    def predict(self, X: np.ndarray):
        # Extend the array with intercept to match weights
        samples, features = X.shape
        X_intercept = np.ones((samples, features + 1))
        X_intercept[:, :-1] = X

        result = np.array([])
        for i in range(samples):
            prediction = np.sign(np.dot(X_intercept[i], self.weights))
            result = np.append(result, prediction)
        return result
