import numpy as np
from perceptron.layer import Layer
from perceptron.optimizer import Optimizer, GradientDescent


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, optimizer: Optimizer = GradientDescent()):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.output_size = output_size
        self.optimizer = optimizer

    def forward(self, input: list[float]):
        self.input = input
        output = np.dot(self.weights, input) + self.bias
        return output

    def backward(self, output_gradient: list[float], learning_rate: float):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        # self.weights -= self.optimizer.adjust(learning_rate, weights_gradient)
        # self.bias -= self.optimizer.adjust(learning_rate, output_gradient)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
