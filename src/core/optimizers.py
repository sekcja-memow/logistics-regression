import abc
import numpy as np


class OptimizerInterface(abc.ABC):
    @abc.abstractmethod
    def optimize(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        optimize method is used to optimize theta array by optimizing algorithm.

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
    >>> optimizer = GradientDescentOptimizer()
    >>> optimizer.optimize(theta, gradient)
    [ 1.2 3.1 1.0 -1.8 ]
    """
    learning_rate: np.float

    def __init__(self, learning_rate: np.float = 0.03):
        self.learning_rate = learning_rate

    def optimize(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return theta - self.learning_rate * gradient

    def __str__(self):
        return f"{type(self).__name__}(learning_rate: {self.learning_rate})"

    def __repr__(self):
        return f"{type(self).__name__}(learning_rate: {self.learning_rate})"


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
        Array used to calculate momentum GD algorithm.
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
        if self.momentum is None:
            self.momentum = np.ones_like(theta)
        self.momentum = self.momentum_rate * self.momentum - self.learning_rate * gradient
        return theta + self.momentum

    def __str__(self):
        return f"{type(self).__name__}(learning_rate: {self.learning_rate}, momentum_rate: {self.momentum_rate})"

    def __repr__(self):
        return f"{type(self).__name__}(learning_rate: {self.learning_rate}, momentum_rate: {self.momentum_rate})"


class AdaGradOptimizer(OptimizerInterface):
    """
    AdaGrad Gradient Descent Optimizer is used to optimize model's parameters array.
    Parameters
    ----------
    learning_rate: np.float, default=0.03
        Learning rate parameter used in optimizer function.
    epsilon: np.float, default=1e-7
        Epsilon is the small ceoffiecent used to avoid dividing by 0.
    Attributes
    ----------
    s: np.ndarray of shape like theta
        Coefficient used by AdaGrad algorithm.
    Examples
    --------
    >>> optimizer = AdaGradOptimizer()
    >>> optimizer.optimize(theta, gradient)
    [ 1.2 3.1 1.0 -1.8 ]
    """
    learning_rate: np.float
    epsilon: np.float
    s: np.ndarray = None

    def __init__(self, learning_rate: np.float = 0.03, epsilon: np.float = 1e-7):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def optimize(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.s is None:
            self.s = np.zeros_like(gradient)
        self.s = self.s + gradient * gradient
        return theta - self.learning_rate * gradient / np.sqrt(self.s + self.epsilon)

    def __str__(self):
        return f"{type(self).__name__}(learning_rate: {self.learning_rate}, epsilon: {self.epsilon})"

    def __repr__(self):
        return f"{type(self).__name__}(learning_rate: {self.learning_rate}, epsilon: {self.epsilon})"


class RMSPropOptimizer(OptimizerInterface):
    """
    RMS Prop Gradient Descent Optimizer is used to optimize model's parameters array.
    RMS Prop is better version of AdaGrad idea.
    Parameters
    ----------
    learning_rate: np.float, default=0.03
        Learning rate parameter used in optimizer function.
    beta: np.float, default=0.09
        Hyperparameter responsible for introducing the exponential distribution
        at the first stage of learning.
    epsilon: np.float, default=1e-7
        Epsilon is the small ceoffiecent used to avoid dividing by 0.
    Attributes
    ----------
    s: np.ndarray of shape like theta
        Coefficient used by AdaGrad algorithm.
    Examples
    --------
    >>> optimizer = RMSPropOptimizer()
    >>> optimizer.optimize(theta, gradient)
    [ 1.2 3.1 1.0 -1.8 ]
    """
    learning_rate: np.float
    beta: np.float
    epsilon: np.float
    s: np.ndarray = None

    def __init__(self, learning_rate: np.float = 0.03, beta: np.float = 0.9, epsilon: np.float = 1e-7):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon

    def optimize(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.s is None:
            self.s = np.zeros_like(gradient)
        self.s = self.beta * self.s + (1 - self.beta) * gradient * gradient
        return theta - self.learning_rate * gradient / np.sqrt(self.s + self.epsilon)

    def __str__(self):
        return f"{type(self).__name__}(learning_rate: {self.learning_rate}, beta: {self.beta}, epsilon: {self.epsilon})"

    def __repr__(self):
        return f"{type(self).__name__}(learning_rate: {self.learning_rate}, beta: {self.beta}, epsilon: {self.epsilon})"


class AdamOptimizer(OptimizerInterface):
    """
    Adam Optimizer (adaptive moment estimation) is used to optimize model's parameters array.
    RMS Prop is better version of AdaGrad idea.
    Parameters
    ----------
    learning_rate: np.float, default=0.03
        Learning rate parameter used in optimizer function.
    beta1: np.float, default=0.9
        Hyperparameter responsible for introducing the exponential distribution
        at the first stage of learning.
    beta1: np.float, default=0.999
        Hyperparameter responsible for introducing the exponential distribution
        at the first stage of learning.
    epsilon: np.float, default=1e-7
        Epsilon is the small ceoffiecent used to avoid dividing by 0.
    Attributes
    ----------
    s: np.ndarray of shape like theta
        Coefficient used by AdaGrad algorithm.
    momentum: np.ndarray of shape like theta
        Array used to calculate momentum GD algorithm.
    t: np.int
        Current timestamp.
    Examples
    --------
    >>> optimizer = AdamOptimizer()
    >>> optimizer.optimize(theta, gradient)
    [ 1.2 3.1 1.0 -1.8 ]
    """
    learning_rate: np.float
    beta1: np.float
    beta2: np.float
    epsilon: np.float
    momentum: np.ndarray = None
    s: np.ndarray = None
    t: np.int = 0

    def __init__(self, learning_rate: np.float = 0.03, beta1: np.float = 0.9, beta2: np.float = 0.999, epsilon: np.float = 1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.s is None:
            self.s = np.zeros_like(gradient)
        if self.momentum is None:
            self.momentum = np.zeros_like(theta)
        self.t += 1
        self.momentum = self.beta1 * self.momentum - (1 - self.beta1) * gradient
        self.s = self.beta2 * self.s + (1 - self.beta2) * gradient * gradient
        self.momentum = self.momentum / (1 - self.beta1 ** self.t)
        self.s = self.s / (1 - self.s ** self.t)
        return theta - self.learning_rate * self.momentum / np.sqrt(self.s + self.epsilon)
    
    def __str__(self):
        return f"{type(self).__name__}(learning_rate: {self.learning_rate}, beta1: {self.beta1}, beta2: {self.beta2}, epsilon: {self.epsilon})"

    def __repr__(self):
        return f"{type(self).__name__}(learning_rate: {self.learning_rate}, beta1: {self.beta1}, beta2: {self.beta2}, epsilon: {self.epsilon})"
