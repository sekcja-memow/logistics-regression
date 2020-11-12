import abc
import numpy as np

from .helpers import sign


class RegularizerInterface(abc.ABC):
    @abc.abstractmethod
    def cost(self, theta: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def gradient(self, theta: np.ndarray) -> np.ndarray:
        pass


class RegularizerL1(RegularizerInterface):
    def __init__(self, alpha: np.float = 0.1):
        self.alpha = alpha

    def cost(self, theta: np.ndarray) -> np.ndarray:
        return self.alpha * np.sum(np.abs(theta))

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        return self.alpha * np.vectorize(sign)(theta)


class RegularizerL2(RegularizerInterface):
    def __init__(self, alpha: np.float = 0.1):
        self.alpha = alpha

    def cost(self, theta: np.ndarray) -> np.ndarray:
        return 0.5 * self.alpha * np.sum(np.power(theta, 2))

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        m = theta.shape[0]
        return (self.alpha / m) * theta


class ElasticNetRegularizer(RegularizerInterface):
    def __init__(self, alpha: np.float = 0.1, beta: np.float = 0.1, gamma: np.float = 0.8):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def cost(self, theta: np.ndarray) -> np.ndarray:
        l1 = self.alpha * np.sum(np.abs(theta))
        l2 = 0.5 * self.beta * np.sum(np.power(theta, 2))
        return self.gamma * l1 + (1 - self.gamma) / 2 * l2

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        m = theta.shape[0]
        l1 = self.alpha * np.vectorize(sign)(theta)
        l2 = (self.beta / m) * theta
        return self.gamma * l1 * (1 - self.gamma) / 2 * l2
