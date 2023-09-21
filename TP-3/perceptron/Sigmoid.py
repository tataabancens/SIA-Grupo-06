import numpy as np
from abc import ABC, abstractmethod


class SigmoidFunction(ABC):
    @abstractmethod
    def eval(x: float) -> float:
        pass

    @abstractmethod
    def eval_derivative(x: float) -> float:
        pass


class SimpleFunction(SigmoidFunction):
    """returns 1 if x >= 0, -1 otherwise."""

    def eval(x: float) -> float:
        return 1 if x == 0 else np.sign(x)

    def eval_derivative(x: float) -> float:
        return 1
