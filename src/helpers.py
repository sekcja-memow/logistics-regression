import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function f(x) = 1 / (1 + exp(-x)).
    Parameters
    ----------
    x: np.ndarray
        Input vector to transform.
    Returns
    -------
    np.ndarray
        Transformated input vector by sigmoid function.
    """
    return 1 / (1 + np.exp(-X))


def sign(theta: int) -> int:
    """
    The sign function.
    Parameters
    ----------
    theta: int
        The sign function argument.
    Returns
    -------
    int
        The sign function result.
    """
    if theta > 0:
        return 1
    elif theta == 0:
        return 0
    return -1
