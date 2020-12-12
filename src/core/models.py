import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score
from .helpers import sigmoid
from .optimizers import OptimizerInterface, RMSPropOptimizer
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

    def __init__(self, optimizer: OptimizerInterface = RMSPropOptimizer(),
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

    def __str__(self):
        return f"{type(self).__name__}(optimizer: {self.optimizer}, regularizer:" \
               f" {self.regularizer}, num_iterations: {self.num_iterations}, threshold: {self.threshold}, " \
               f"fit_intercept: {self.fit_intercept})"

    def __repr__(self):
        return f"{type(self).__name__}(optimizer: {self.optimizer}, regularizer:" \
               f" {self.regularizer}, num_iterations: {self.num_iterations}, threshold: {self.threshold}, " \
               f"fit_intercept: {self.fit_intercept})"

    def get_params(self, deep=False):
        return {
                "optimizer": self.optimizer,
                "regularizer": self.regularizer,
                "num_iterations": self.num_iterations,
                "threshold": self.threshold,
                "fit_intercept": self.fit_intercept
                }

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
            return grad + self.regularizer.gradient(theta)
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

        for epoch in range(1, self.num_iterations+1):
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
        try:
            return TP / (TP + FP)
        except ZeroDivisionError:
            return np.inf


class SVMClassifier:
    """
    TODO: docs
    """
    def __init__(self, max_epochs: int = 1024, regularization_strength: int = 10000,
                 learning_rate: float = 0.000001, cost_threshold: float = 0.01,
                 fit_intercept: bool = True, verbose: bool = True):
        # TODO: reformat interface
        self.max_epochs = max_epochs
        self.regularization_strength = regularization_strength
        self.learning_rate = learning_rate
        self.cost_threshold = cost_threshold

        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.weights = []

    def __str__(self):
        return f"{type(self).__name__}(max_epochs: {self.max_epochs}, regularization_strength:" \
               f" {self.regularization_strength}, learning_rate: {self.learning_rate}, cost_threshold: " \
               f"{self.cost_threshold}, fit_intercept: {self.fit_intercept})"

    def __repr__(self):
        return f"{type(self).__name__}(max_epochs: {self.max_epochs}, regularization_strength:" \
               f" {self.regularization_strength}, learning_rate: {self.learning_rate}, cost_threshold: " \
               f"{self.cost_threshold}, fit_intercept: {self.fit_intercept})"

    def cost(self, W, X, Y) -> float:
        # calculate hinge loss
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = self.regularization_strength * (np.sum(distances) / N)

        # calculate cost
        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost

    def gradient(self, W, X_batch, Y_batch) -> np.ndarray:
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
        if self.fit_intercept:
            X = self.add_intercept(X)
        # Temp variables
        weights = np.zeros(X.shape[1])
        nth = 0
        prev_cost = float("inf")
        # stochastic gradient descent
        for epoch in range(1, self.max_epochs):
            # shuffle to prevent repeating update cycles
            Xs, ys = shuffle(X, y)
            for ind, x in enumerate(Xs):
                ascent = self.gradient(weights, x, ys[ind])
                weights = weights - (self.learning_rate * ascent)

            # convergence check on 2^nth epoch
            if epoch == 2 ** nth or epoch == self.max_epochs - 1:
                cost = self.cost(weights, X, y)
                if self.verbose:
                    print("Epoch is: {} and Cost is: {}".format(epoch, cost))
                # stoppage criterion
                if abs(prev_cost - cost) < self.cost_threshold * prev_cost:
                    self.weights = weights
                    return
                prev_cost = cost
                nth += 1
        self.weights = weights

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
        result = np.array([])
        for i in range(X.shape[0]):
            prediction = np.sign(np.dot(X[i], self.weights))
            result = np.append(result, prediction)
        return result

    @staticmethod
    def evaluate(y_test, y_pred) -> None:
        """
        Check the accuaracy of the model.
        Parameters
        ----------
        y_test: np.ndarray
            Test result vector.
        y_pred: np.ndarray
            Vector predicted by classifier.
        """
        print("accuracy on test dataset: {}".format(
            accuracy_score(y_test, y_pred)))
        print("recall on test dataset: {}".format(
            recall_score(y_test, y_pred)))
        print("precision on test dataset: {}".format(
            recall_score(y_test, y_pred)))

    def add_intercept(self, X) -> np.ndarray:
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
