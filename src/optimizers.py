import abc
import numpy as np


class OptimizerInterface(abc.ABC):
    @abc.abstractmethod
    def optimize(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        pass


class GradientDescentOptimizer(OptimizerInterface):
    """
    Gradient Descent Optimizer is used to optimize model's parameters array.
    Parameters
    ----------
    learning_rate: np.float, default=0.03
        Learning rate parameter used in optimizer function.
    Examples
    --------
    >>> optimizer = MomentumGradientDescentOptimizer()
    >>> optimizer.optimize(theta, gradient)
    [ 1.2 3.1 1.0 -1.8 ]
    """
    learning_rate: np.float

    def __init__(self, learning_rate: np.float = 0.03):
        self.learning_rate = learning_rate

    def optimize(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        optimize method is used to optimize theta array by GD algorithm

        Parameters
        ----------
        theta: np.ndarray
            Input array to be transformed by algorithm (shape [n x m])
        gradient: np.ndarray
            Gradient array used by algorithm to optimize theta (shape [n x m])
        Returns
        -------
        np.ndarray
            Optimized theta array.
        """
        return theta - self.learning_rate * gradient


class MomentumGradientDescentOptimizer(OptimizerInterface):
    """
    Momentum Gradient Descent Optimizer is used to optimize model's parameters array.
    Parameters
    ----------
    learning_rate: np.float, default=0.03
        Learning rate parameter used in optimizer function.
    momentum_rate: np.float, default=0.9
        Momentum rate parameter used in momentum optimizer algorithm.
        Momentum rate must be from interval from 0 to 1.
    Attributes
    ----------
    momentum: np.ndarray of shape like theta
        Coefficient used to calculate momentum GD algorithm.
    Examples
    --------
    >>> optimizer = MomentumGradientDescentOptimizer()
    >>> optimizer.optimize(theta, gradient)
    [ 1.2 3.1 1.0 -1.8 ]
    """
    learning_rate: np.float
    momentum_rate: np.float
    momentum: np.ndarray = None

    def __init__(self, learning_rate: np.float = 0.03, momentum_rate: np.float = 0.9):
        self.learning_rate = learning_rate
        if momentum_rate < 0 or momentum_rate > 1:
            raise ValueError("momentum_rate value must be from interval from 0 to 1.")
        self.momentum_rate = momentum_rate

    def optimize(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        optimize method is used to optimize theta array by Momentum GD algorithm

        Parameters
        ----------
        theta: np.ndarray
            Input array to be transformed by algorithm (shape [n x m])
        gradient: np.ndarray
            Gradient array used by algorithm to optimize theta (shape [n x m])
        Returns
        -------
        np.ndarray
            Optimized theta array.
        """
        if self.momentum is None:
            self.momentum = np.ones_like(theta)
        self.momentum = self.momentum_rate * self.momentum - self.learning_rate * gradient
        return theta + self.momentum
