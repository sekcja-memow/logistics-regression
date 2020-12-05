import abc
import numpy as np

from .helpers import sign


class RegularizerInterface(abc.ABC):
    @abc.abstractmethod
    def cost(self, theta: np.ndarray) -> np.ndarray:
        """
        cost method is used to calculate additional cost element that limits the model.

        Parameters
        ----------
        theta: np.ndarray
            Input array to be transformed by algorithm (shape [n x m])
        Returns
        -------
        np.ndarray
            Additional cost element.
        """
        pass

    @abc.abstractmethod
    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        gradient method is used to calculate additional gradient element that limits the model.

        Parameters
        ----------
        theta: np.ndarray
            Input array to be transformed by algorithm (shape [n x m])
        Returns
        -------
        np.ndarray
            Additional gradient element.
        """
        pass


class LassoRegularizer(RegularizerInterface):
    """
    Regularizer L1 (Lasso Regularizer) is used to limit model's cost and gradient functions.

    Read more: https://en.wikipedia.org/wiki/Regularization_(mathematics)

    Parameters
    ----------
    alpha: np.float, default=0.1
        alpha is the L1 regularization factor.
    Examples
    --------
    >>> regularizer = RegularizerL1()
    >>> regularizer.cost(theta)
    [ 1.2 3.1 1.0 -1.8 ]
    >>> regularizer.gradient(theta)
    [ 0.2 4.1 1.3 -2.4 ]
    """

    def __init__(self, alpha: np.float = 0.1):
        self.alpha = alpha

    def __str__(self):
        return f'{type(self).__name__}(alpha: {self.alpha})'

    def __repr__(self):
        return f'{type(self).__name__}(alpha: {self.alpha})'

    def cost(self, theta: np.ndarray) -> np.ndarray:
        return self.alpha * np.sum(np.abs(theta))

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        return self.alpha * np.vectorize(sign)(theta)


class RidgeRegularizer(RegularizerInterface):
    """
    Regularizer L2 (Ridge Regularizer) is used to limit model's cost and gradient functions.

    Read more: https://en.wikipedia.org/wiki/Regularization_(mathematics)

    Parameters
    ----------
    alpha: np.float, default=0.1
        alpha is the L2 regularization factor.
    Examples
    --------
    >>> regularizer = RegularizerL2()
    >>> regularizer.cost(theta)
    [ 1.2 3.1 1.0 -1.8 ]
    >>> regularizer.gradient(theta)
    [ 0.2 4.1 1.3 -2.4 ]
    """

    def __init__(self, alpha: np.float = 0.1):
        self.alpha = alpha

    def __str__(self):
        return f'{type(self).__name__}(alpha: {self.alpha})'

    def __repr__(self):
        return f'{type(self).__name__}(alpha: {self.alpha})'

    def cost(self, theta: np.ndarray) -> np.ndarray:
        return 0.5 * self.alpha * np.sum(np.power(theta, 2))

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        m = theta.shape[0]
        return (self.alpha / m) * theta


class ElasticNetRegularizer(RegularizerInterface):
    """
    ElasticNetRegularizer is the combination of L1 and L2 regularizers, it is used to limit model's cost and gradient.
    functions.

    Read more: https://en.wikipedia.org/wiki/Regularization_(mathematics)

    Parameters
    ----------
    alpha: np.float, default=0.1
        alpha is L1 the regularization factor.
    beta: np.float, default=0.1
        alpha is L2 the regularization factor.
    gamma: np.float, default=0.8
        gamma is the mixing parameter between ridge and lasso.
    Examples
    --------
    >>> regularizer = RegularizerL1()
    >>> regularizer.cost(theta)
    [ 1.2 3.1 1.0 -1.8 ]
    >>> regularizer.gradient(theta)
    [ 0.2 4.1 1.3 -2.4 ]
    """

    def __init__(self, alpha: np.float = 0.1, beta: np.float = 0.1, gamma: np.float = 0.8):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __str__(self):
        return f'{type(self).__name__}(alpha: {self.alpha}, beta: {self.beta}, gamma: {self.gamma})'

    def __repr__(self):
        return f'{type(self).__name__}(alpha: {self.alpha}, beta: {self.beta}, gamma: {self.gamma})'

    def cost(self, theta: np.ndarray) -> np.ndarray:
        l1 = self.alpha * np.sum(np.abs(theta))
        l2 = 0.5 * self.beta * np.sum(np.power(theta, 2))
        return self.gamma * l1 + (1 - self.gamma) / 2 * l2

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        m = theta.shape[0]
        l1 = self.alpha * np.vectorize(sign)(theta)
        l2 = (self.beta / m) * theta
        return self.gamma * l1 * (1 - self.gamma) / 2 * l2
