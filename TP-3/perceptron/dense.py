import numpy as np
from layer import Layer


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input: list[float]):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient: list[float], learning_rate: float):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
