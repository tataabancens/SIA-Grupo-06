import math
from perceptron.LinealPerceptron import LinealPerceptron
from enum import Enum


class PerceptronType(Enum):
    Lineal = "Lineal"
    NonLinealTanh = "NonLinealTanh"
    NonLinealSigmoid = "NonLinealSigmoid"


class NonLinealTanh(LinealPerceptron):

    def __init__(self, size, learning_rate, *args, **kwargs):
        super().__init__(size, learning_rate, *args, **kwargs)
        self.B = kwargs["B"]

    def activation(self, excitement: float) -> float:
        return math.tanh(self.B * excitement)

    def activation_derivative(self, excitement: float) -> float:
        return self.B * (1 - math.pow(self.activation(excitement), 2))


class NonLinealSigmoid(NonLinealTanh):
    def activation(self, excitement: float) -> float:
        return 1.0 / (1.0 + math.exp(-2 * self.B * excitement))

    def activation_derivative(self, excitement: float) -> float:
        activation = self.activation(excitement)
        return 2 * self.B * activation * (1 - activation)
