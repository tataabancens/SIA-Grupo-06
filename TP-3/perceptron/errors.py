import numpy as np
from abc import ABC, abstractmethod

class ErrorFunction(ABC):
    @abstractmethod
    def eval(y_true, y_pred) -> float:
        pass

    @abstractmethod
    def eval_derivative(y_true, y_pred) -> float:
        pass

class BinaryCrossEntropy(ErrorFunction):
    def eval(self, y_true, y_pred) -> float:
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    def eval_derivative(self, y_true, y_pred) -> float:
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)




class MeanSquared(ErrorFunction):
    def eval(self, y_true, y_pred) -> float:
        return np.mean(np.power(y_true - y_pred, 2))

    def eval_derivative(self, y_true, y_pred) -> float:
        return 2 * (y_pred - y_true) / np.size(y_true)
